import cv2
import time

# Buka kamera (index 0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka. Coba index 1 atau 2.")

frames, t0 = 0, time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frames += 1

    # Hitung FPS setiap 1 detik
    if time.time() - t0 >= 1.0:
        fps = frames / (time.time() - t0)
        cv2.setWindowTitle("Preview", f"Preview (FPS ~ {fps:.1f})")
        frames, t0 = 0, time.time()

    cv2.imshow("Preview", frame)

    # Tekan Q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
