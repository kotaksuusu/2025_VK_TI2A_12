import cv2
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector

# --- Indeks mata kiri (berdasarkan model MediaPipe) ---
# vertikal (159,145), horizontal (33,133)
L_TOP, L_BOTTOM, L_LEFT, L_RIGHT = 159, 145, 33, 133

# --- Fungsi jarak Euclidean antar titik ---
def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# --- Inisialisasi kamera ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    raise RuntimeError("❌ Kamera tidak bisa dibuka.")

# --- Inisialisasi FaceMeshDetector ---
detector = FaceMeshDetector(
    staticMode=False,
    maxFaces=2,
    minDetectionCon=0.5,
    minTrackCon=0.5
)

# --- Variabel untuk menghitung kedipan ---
blink_count = 0
closed_frames = 0
CLOSED_FRAMES_THRESHOLD = 3   # jumlah frame berturut-turut dianggap kedipan
EYE_AR_THRESHOLD = 0.20       # ambang EAR: makin kecil = mata tertutup
is_closed = False

# --- Loop utama ---
while True:
    ok, img = cap.read()
    if not ok:
        print("⚠️ Gagal membaca frame dari kamera.")
        break

    # Deteksi face mesh
    img, faces = detector.findFaceMesh(img, draw=True)

    if faces:
        # Ambil wajah pertama
        face = faces[0]

        # Hitung jarak vertikal & horizontal pada mata kiri
        v = dist(face[L_TOP], face[L_BOTTOM])
        h = dist(face[L_LEFT], face[L_RIGHT])
        ear = v / (h + 1e-8)

        # Tampilkan nilai EAR
        cv2.putText(img, f"EAR (L): {ear:.3f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # --- Logika deteksi kedipan ---
        if ear < EYE_AR_THRESHOLD:
            closed_frames += 1
            if closed_frames >= CLOSED_FRAMES_THRESHOLD and not is_closed:
                blink_count += 1
                is_closed = True
        else:
            closed_frames = 0
            is_closed = False

        # Tampilkan jumlah kedipan
        cv2.putText(img, f"Blink Count: {blink_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Tampilkan hasil
    cv2.imshow("FaceMesh + Blink Detection", img)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Bersihkan resource ---
cap.release()
cv2.destroyAllWindows()
