import time
import threading
import cv2
import mss
import numpy as np
import pyautogui

SHOW_DEBUG = True

SCAN_HEIGHT = 130
MERGE_DIST  = 80
FRAME_LIMIT = 0.88

SPEED_BASE  = 6.0
SPEED_GROW  = 0.008
SPEED_CAP   = 35.0

REACT = {
    "small":  (0.95, 0.10),
    "medium": (1.00, 0.15),
    "large":  (1.05, 0.22),
    "group":  (0.80, 0.27),
    "bird_j": (1.00, 0.13),
} 
 
DIST_BASE    = 90
DIST_K1      = 3.5
DIST_K2      = 0.18
DIST_MIN     = 80
DIST_MAX     = 500

HI_BIRD_K    = 1.3
MD_BIRD_K    = 0.55
BIRD_MAX_K   = 2.5

DUCK_RANGE   = 400
DUCK_K       = 1.5
DUCK_MIN     = 0.20
DUCK_MAX     = 1.10
DUCK_LOCKOUT = 0.08
AIR_EXTRA    = 0.40
OVER_CONFIRM = 3


def elapsed_speed(t0: float) -> float:
    secs = time.perf_counter() - t0
    return min(SPEED_BASE + SPEED_GROW * secs * 60, SPEED_CAP)


def scan_for_dino(gray):
    h, w = gray.shape
    half = gray[:, : w // 2].copy()
    _, mask = cv2.threshold(half, 100, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, score = None, 0
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        if abs((y + ch) - h) <= 20 and 20 <= ch <= 120 and 20 <= cw <= 150:
            if cw * ch > score:
                score, best = cw * ch, (x, y, cw, ch)
    return best


def locate_game(sct):
    mon  = sct.monitors[1]
    img  = cv2.cvtColor(np.array(sct.grab(mon)), cv2.COLOR_BGRA2GRAY)
    H, W = img.shape

    ground, peak = None, 0
    for row in range(H // 3, H - 20):
        dark = int(np.sum(img[row, W // 4: 3 * W // 4] < 100))
        if dark > peak and dark > W // 8:
            peak, ground = dark, row

    if ground is None:
        return None

    top   = max(0, ground - SCAN_HEIGHT)
    strip = img[top: ground + 5, :]
    dino  = scan_for_dino(strip)
    if dino is None:
        return None

    dx, dy, dw, dh = dino
    return {
        "region":   {"left": 0, "top": top, "width": mon["width"],
                     "height": SCAN_HEIGHT + 10, "mon": 1},
        "dino_rx":  dx + dw,
        "ground_y": ground - top,
        "dino_h":   dh,
    }


def await_game(sct):
    print("[INFO] Поиск игры... Откройте браузер с динозавром.")
    deadline = time.time() + 30
    while time.time() < deadline:
        z = locate_game(sct)
        if z:
            print(f"[INFO] Найдено: dino_rx={z['dino_rx']} ground_y={z['ground_y']} dino_h={z['dino_h']}")
            return z
        time.sleep(0.5)
    raise RuntimeError("Игра не обнаружена за 30 секунд")


def tag_cactus(w, h):
    if w >= 70:      return "group"
    if h <= 35:      return "small"
    if h <= 55:      return "medium"
    return "large"


def tag_bird(gap, dino_h):
    if gap >= dino_h * HI_BIRD_K: return "bird_hi"
    if gap >= dino_h * MD_BIRD_K: return "bird_duck"
    return "bird_j"


def looks_like_ui(bw, bh):
    return bh > 35 and (bw / max(bh, 1)) < 0.8


def find_obstacles(gray, dino_rx, ground_y, thr, dino_h):
    W = gray.shape[1]
    _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
    mask[ground_y:, :]              = 0
    mask[: max(0, ground_y - 115), :] = 0
    mask[:, : dino_rx + 20]         = 0
    mask[:, int(W * FRAME_LIMIT):]  = 0

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    on_ground, in_air = [], []
    bird_gap_max = int(dino_h * BIRD_MAX_K)

    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        if bh < 10 or bw * bh < 60:
            continue
        gap = ground_y - (y + bh)
        if gap <= 8:
            if bh <= 110:
                on_ground.append((x, y, bw, bh))
        else:
            if gap > bird_gap_max or looks_like_ui(bw, bh):
                continue
            in_air.append((x, y, bw, bh, gap))

    result = []

    if on_ground:
        on_ground.sort(key=lambda b: b[0])
        merged = [list(on_ground[0])]
        for bx, by, bw, bh in on_ground[1:]:
            px, py, pw, ph = merged[-1]
            if bx <= px + pw + MERGE_DIST:
                merged[-1] = [px, min(py, by),
                               max(px + pw, bx + bw) - px,
                               max(py + ph, by + bh) - min(py, by)]
            else:
                merged.append([bx, by, bw, bh])
        for x, y, cw, ch in merged:
            d = x - dino_rx
            if d <= 0: continue
            result.append((d, tag_cactus(cw, ch), cw, ch, ground_y - (y + ch)))

    for x, y, bw, bh, gap in in_air:
        d = x - dino_rx
        if d <= 0: continue
        result.append((d, tag_bird(gap, dino_h), bw, bh, gap))

    return sorted(result, key=lambda r: r[0])


def duck_duration(dist, speed):
    pps = max(speed * 100.0, 200.0)
    return float(np.clip(dist / pps * DUCK_K, DUCK_MIN, DUCK_MAX))


def detect_game_over(gray, ground_y):
    y1 = max(0, ground_y - 80)
    y2 = max(1, ground_y - 25)
    if y2 <= y1:
        return False
    W = gray.shape[1]
    roi = gray[y1:y2, W // 3: 2 * W // 3]
    if roi.size == 0:
        return False
    _, bw = cv2.threshold(roi, 140, 255, cv2.THRESH_BINARY_INV)
    return (np.sum(bw) / (bw.size * 255 + 1e-8)) > 0.12


class BotState:
    def __init__(self):
        self.lock  = threading.Lock()
        self.frame = None
        self.stop  = False


def run_bot(state: BotState):
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE    = 0

    with mss.mss() as sct:
        zone     = await_game(sct)
        reg      = zone["region"]
        dino_rx  = zone["dino_rx"]
        ground_y = zone["ground_y"]
        dino_h   = zone["dino_h"]

        print("[INFO] Старт через 3 секунды...")
        time.sleep(3)
        pyautogui.press("space")
        time.sleep(1)

        raw = cv2.cvtColor(np.array(sct.grab(reg)), cv2.COLOR_BGRA2GRAY)
        hist = cv2.calcHist([raw], [0], None, [256], [0, 256]).flatten()
        hist[:150] = 0
        thr = max(int(np.argmax(hist)) - 40, 100)
        print(f"[INFO] Порог бинаризации: {thr}")

        t_start      = time.perf_counter()
        jump_up      = 0.0
        land_at      = 0.0
        duck_up      = 0.0
        free_at      = 0.0
        jumping      = False
        ducking      = False
        over_cd      = 0.0
        over_streak  = 0
        last_act     = "—"

        while not state.stop:
            t0  = time.perf_counter()
            now = t0

            gray  = cv2.cvtColor(np.array(sct.grab(reg)), cv2.COLOR_BGRA2GRAY)
            speed = elapsed_speed(t_start)

            if jumping and now >= jump_up:
                pyautogui.keyUp("space")
                jumping = False

            if ducking and now >= duck_up:
                pyautogui.keyUp("down")
                ducking = False
                free_at = max(free_at, now + DUCK_LOCKOUT)

            if now > over_cd:
                if detect_game_over(gray, ground_y):
                    over_streak += 1
                else:
                    over_streak = 0

                if over_streak >= OVER_CONFIRM:
                    over_streak = 0
                    if jumping: pyautogui.keyUp("space"); jumping = False
                    if ducking: pyautogui.keyUp("down");  ducking = False
                    print("[BOT] Game over — перезапуск")
                    time.sleep(0.5)
                    pyautogui.press("space")
                    time.sleep(1.0)
                    t_start  = time.perf_counter()
                    over_cd  = now + 3.0
                    free_at  = now + 3.0
                    land_at  = 0.0
                    last_act = "RESTART"
                    continue
            else:
                over_streak = 0

            obstacles = find_obstacles(gray, dino_rx, ground_y, thr, dino_h)
            base_d = np.clip(DIST_BASE + DIST_K1 * speed + DIST_K2 * speed ** 2,
                             DIST_MIN, DIST_MAX)

            airborne = now < land_at

            duck_cand = None
            if not airborne:
                for item in obstacles:
                    if item[0] > DUCK_RANGE: break
                    if item[1] == "bird_duck":
                        duck_cand = item
                        break

            if duck_cand and not jumping:
                d, k, ow, oh, gap = duck_cand
                hold = duck_duration(d, speed)
                if ducking:
                    new_end = now + hold
                    if new_end > duck_up:
                        duck_up = new_end
                        free_at = duck_up + DUCK_LOCKOUT
                else:
                    pyautogui.keyDown("down")
                    ducking  = True
                    duck_up  = now + hold
                    free_at  = duck_up + DUCK_LOCKOUT
                    last_act = f"DUCK d={d} hold={hold:.2f}s spd={speed:.1f}"
                    print(f"[BOT] {last_act}")

            elif obstacles and now >= free_at and not ducking:
                dist, kind, ow, oh, gap = obstacles[0]
                if kind in REACT and kind != "bird_hi":
                    mult, hold_t = REACT[kind]
                    trigger = int(np.clip(base_d * mult, DIST_MIN, DIST_MAX))
                    if dist <= trigger and not jumping:
                        pyautogui.keyDown("space")
                        jumping  = True
                        jump_up  = now + hold_t
                        land_at  = now + hold_t + AIR_EXTRA
                        free_at  = land_at
                        last_act = f"JUMP {kind} d={dist} trig={trigger} spd={speed:.1f}"
                        print(f"[BOT] {last_act}")

            if SHOW_DEBUG:
                vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                h, w = vis.shape[:2]

                cv2.line(vis, (0, ground_y), (w, ground_y), (180, 105, 255), 1)
                cv2.line(vis, (dino_rx, 0), (dino_rx, h), (180, 105, 255), 2)

                for item in obstacles[:6]:
                    d, k, ow, oh, gap = item
                    ox   = dino_rx + d
                    ty   = ground_y - oh
                    col  = (180, 0, 180)
                    cv2.rectangle(vis, (ox, ty), (ox + ow, ground_y), col, 2)
                    cv2.putText(vis, f"{k} {d}", (ox, max(ty - 3, 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, col, 1)

                info = f"SPD:{speed:.1f} base={int(base_d)} | "
                if jumping:  info += f"[JUMP {max(jump_up - now, 0):.2f}s] "
                elif airborne: info += f"[AIR {max(land_at - now, 0):.2f}s] "
                elif ducking:  info += f"[DUCK {max(duck_up - now, 0):.2f}s] "
                elif now < free_at: info += f"[WAIT {free_at - now:.2f}s] "
                else:              info += "[READY] "
                info += last_act

                cv2.putText(vis, info, (10, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 220), 1)

                with state.lock:
                    state.frame = vis

            time.sleep(max(0.001, 0.010 - (time.perf_counter() - t0)))

    if jumping: pyautogui.keyUp("space")
    if ducking: pyautogui.keyUp("down")


def main():
    state  = BotState()
    worker = threading.Thread(target=run_bot, args=(state,), daemon=True)
    worker.start()

    if SHOW_DEBUG:
        print("[GUI] Нажмите Q для выхода")
        while worker.is_alive():
            with state.lock:
                frame = state.frame
            if frame is not None:
                cv2.imshow("DinoBot", frame)
            if cv2.waitKey(8) & 0xFF == ord('q'):
                state.stop = True
                break
        cv2.destroyAllWindows()
    else:
        worker.join()

  
if __name__ == "__main__":
    main()
