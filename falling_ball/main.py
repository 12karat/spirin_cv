import sys
import math
import threading
import time
import cv2
import numpy as np

GRAVITY        = 0.4
BOUNCE_DAMPING = 0.55
FRICTION       = 0.988
BALL_RADIUS    = 18
MIN_BOUNCE_VY  = 1.5

PHYSICS_FPS = 120
CAMERA_FPS  = 30

COLOR_BALL_FILL   = (40,  80,  255)
COLOR_BALL_SHINE  = (200, 220, 255)
COLOR_PLATFORM    = (50,  220, 80)
COLOR_TRAIL       = (80,  140, 255)
COLOR_HUD_TEXT    = (200, 200, 200)
COLOR_START_CROSS = (0,   200, 255)

class Ball:
    def __init__(self, x: float, y: float):
        self.radius  = BALL_RADIUS
        self.start_x = float(x)
        self.start_y = float(y)
        self._lock   = threading.Lock()
        self._reset_state(x, y)

    def _reset_state(self, x, y):
        self.x        = float(x)
        self.y        = float(y)
        self.vx       = 0.0
        self.vy       = 0.0
        self.active   = False
        self.on_ground = False
        self.trail: list[tuple[int, int]] = []

    def reset(self, x: float, y: float):
        with self._lock:
            self._reset_state(x, y)

    def start(self):
        with self._lock:
            self.active = True
            self.vy = 1.0

    def update(self, platforms: list, screen_w: int, screen_h: int):
        with self._lock:
            if not self.active:
                return

            self.vy += GRAVITY
            self.x  += self.vx
            self.y  += self.vy
            self.vx *= FRICTION
            self.on_ground = False

            self.trail.append((int(self.x), int(self.y)))
            if len(self.trail) > 25:
                self.trail.pop(0)

            for platform in platforms:
                self._resolve_platform(platform)

            if (self.y - self.radius > screen_h + 60 or
                    self.x < -120 or self.x > screen_w + 120):
                self._reset_state(self.start_x, self.start_y)

    def _resolve_platform(self, platform):
        p1     = np.array(platform[0], dtype=float)
        p2     = np.array(platform[1], dtype=float)
        center = np.array([self.x, self.y], dtype=float)

        seg     = p2 - p1
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-6:
            return

        seg_dir = seg / seg_len
        normal  = np.array([-seg_dir[1], seg_dir[0]])

        t        = float(np.clip(np.dot(center - p1, seg_dir), 0.0, seg_len))
        closest  = p1 + t * seg_dir
        dist_vec = center - closest
        dist     = float(np.linalg.norm(dist_vec))

        if dist < self.radius and dist > 1e-6:
            if np.dot(dist_vec, normal) < 0:
                normal = -normal

            vel   = np.array([self.vx, self.vy])
            vel_n = float(np.dot(vel, normal))
            if vel_n < 0:
                self.x += normal[0] * (self.radius - dist)
                self.y += normal[1] * (self.radius - dist)

                vel_normal  = vel_n * normal
                vel_tangent = vel - vel_normal
                self.vx = vel_tangent[0] - vel_normal[0] * BOUNCE_DAMPING
                self.vy = vel_tangent[1] - vel_normal[1] * BOUNCE_DAMPING

                if abs(self.vy) < MIN_BOUNCE_VY and normal[1] < -0.5:
                    self.vy = 0.0
                    self.on_ground = True

    @property
    def pos(self) -> tuple[int, int]:
        return int(self.x), int(self.y)


class PlatformDetector:
    def __init__(
        self,
        min_line_length: int  = 80,
        max_line_gap: int     = 20,
        min_votes: int        = 50,
        max_angle: float      = 60.0,
    ):
        self.min_line_length = min_line_length
        self.max_line_gap    = max_line_gap
        self.min_votes       = min_votes
        self.max_angle       = max_angle

    def detect(self, frame: np.ndarray) -> list:
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.min_votes,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        platforms = []
        if lines is None:
            return platforms
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            angle = abs(math.degrees(math.atan2(abs(dy), max(abs(dx), 1))))
            if angle <= self.max_angle:
                platforms.append([(int(x1), int(y1)), (int(x2), int(y2))])
        return platforms

class Renderer:
    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height
        self._canvas = np.zeros((height, width, 3), dtype=np.uint8)

    def draw(self, ball: Ball, platforms: list) -> np.ndarray:
        out = self._canvas.copy()
        self._draw_platforms(out, platforms)
        self._draw_trail(out, ball)
        if ball.active:
            self._draw_ball(out, ball)
        else:
            self._draw_start_marker(out, ball)
        self._draw_hud(out, ball, len(platforms))
        return out

    def _draw_platforms(self, img, platforms):
        for p in platforms:
            cv2.line(img, p[0], p[1], COLOR_PLATFORM, 3, cv2.LINE_AA)
            cv2.circle(img, p[0], 4, COLOR_PLATFORM, -1, cv2.LINE_AA)
            cv2.circle(img, p[1], 4, COLOR_PLATFORM, -1, cv2.LINE_AA)

    def _draw_trail(self, img, ball: Ball):
        trail = ball.trail
        n = len(trail)
        for i in range(1, n):
            alpha = i / n
            color = tuple(int(c * alpha) for c in COLOR_TRAIL)
            cv2.line(img, trail[i-1], trail[i], color,
                     max(1, int(3 * alpha)), cv2.LINE_AA)

    def _draw_ball(self, img, ball: Ball):
        r = ball.radius
        cx, cy = ball.pos
        off = max(3, r // 4)
        cv2.ellipse(img, (cx + off, cy + off), (r, r // 2),
                    0, 0, 360, (30, 30, 30), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), r, COLOR_BALL_FILL, -1, cv2.LINE_AA)
        cv2.circle(img, (cx - r//3, cy - r//3), max(2, r//3),
                   COLOR_BALL_SHINE, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), r, (20, 40, 180), 2, cv2.LINE_AA)

    def _draw_start_marker(self, img, ball: Ball):
        cx, cy = int(ball.start_x), int(ball.start_y)
        s = ball.radius + 6
        cv2.line(img, (cx-s, cy), (cx+s, cy), COLOR_START_CROSS, 2, cv2.LINE_AA)
        cv2.line(img, (cx, cy-s), (cx, cy+s), COLOR_START_CROSS, 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), ball.radius, COLOR_START_CROSS, 2, cv2.LINE_AA)
        cv2.putText(img, "SPACE to start",
                    (cx - 60, cy - ball.radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    COLOR_START_CROSS, 1, cv2.LINE_AA)

    def _draw_hud(self, img, ball: Ball, platform_count: int):
        lines = [
            f"Platforms: {platform_count}",
            f"Vx:{ball.vx:+.1f}  Vy:{ball.vy:+.1f}",
            "SPACE: reset   Q/ESC: quit",
        ]
        for i, text in enumerate(lines):
            cv2.putText(img, text, (10, 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        COLOR_HUD_TEXT, 1, cv2.LINE_AA)

class SharedState:
    def __init__(self):
        self.lock      = threading.Lock()
        self.platforms: list = []
        self.running   = True

def camera_thread(cap, detector: PlatformDetector, state: SharedState):
    interval = 1.0 / CAMERA_FPS
    while state.running:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            state.running = False
            break
        frame     = cv2.flip(frame, 1)
        platforms = detector.detect(frame)
        with state.lock:
            state.platforms = platforms
        wait = interval - (time.perf_counter() - t0)
        if wait > 0:
            time.sleep(wait)


def physics_thread(ball: Ball, state: SharedState, w: int, h: int):
    interval = 1.0 / PHYSICS_FPS
    while state.running:
        t0 = time.perf_counter()
        with state.lock:
            platforms = list(state.platforms)
        ball.update(platforms, w, h)
        wait = interval - (time.perf_counter() - t0)
        if wait > 0:
            time.sleep(wait)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: не удаётся открыть камеру.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            break
    if not ret:
        print("Ошибка: нет кадра с камеры.")
        sys.exit(1)

    h, w    = frame.shape[:2]
    start_x = w // 2
    start_y = 40

    ball     = Ball(start_x, start_y)
    detector = PlatformDetector()
    renderer = Renderer(w, h)
    state    = SharedState()

    threading.Thread(target=camera_thread,
                     args=(cap, detector, state), daemon=True).start()
    threading.Thread(target=physics_thread,
                     args=(ball, state, w, h), daemon=True).start()

    cv2.namedWindow("Falling Ball AR", cv2.WINDOW_NORMAL)
    print("=" * 50)
    print("  Falling Ball AR  |  чёрный фон + шарик")
    print("  ПРОБЕЛ — старт/сброс    Q/ESC — выход")
    print("=" * 50)

    while state.running:
        with state.lock:
            platforms = list(state.platforms)

        output = renderer.draw(ball, platforms)
        cv2.imshow("Falling Ball AR", output)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord(' '):
            ball.reset(start_x, start_y)
            ball.start()

    state.running = False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()