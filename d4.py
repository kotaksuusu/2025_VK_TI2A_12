import cv2
from cvzone.HandTrackingModule import HandDetector

# --- Inisialisasi kamera ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW = buka webcam lebih cepat di Windows
cap.set(3, 640)  # lebar frame
cap.set(4, 480)  # tinggi frame

if not cap.isOpened():
    raise RuntimeError("❌ Kamera tidak bisa dibuka.")

# --- Inisialisasi detektor tangan ---
detector = HandDetector(
    staticMode=False,     # deteksi berkelanjutan (real-time)
    maxHands=1,           # hanya 1 tangan untuk efisiensi
    modelComplexity=0,    # model lebih ringan, cukup akurat
    detectionCon=0.5,     # ambang deteksi awal
    minTrackCon=0.5       # ambang pelacakan saat tangan bergerak
)

# --- Loop utama ---
while True:
    ok, img = cap.read()
    if not ok:
        print("⚠️ Gagal membaca frame kamera.")
        break

    # Deteksi tangan dan landmark
    hands, img = detector.findHands(img, draw=True, flipType=True)  # flipType=True biar mirror

    if hands:
        hand = hands[0]  # ambil tangan pertama
        fingers = detector.fingersUp(hand)  # list [1,1,0,0,0] misalnya
        count = sum(fingers)

        # Tampilkan jumlah jari dan status biner
        cv2.putText(img, f"Fingers: {count}  {fingers}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Tampilkan hasil di jendela
    cv2.imshow("Hand Tracking - Finger Count", img)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Bersihkan resource ---
cap.release()
cv2.destroyAllWindows()
