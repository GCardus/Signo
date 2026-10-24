import cv2
import mediapipe as mp
import csv
import os
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Provar diferents índexs de càmera per si el 0 no funciona
cap = None
for idx in [0, 1, 2]:
    test = cv2.VideoCapture(idx)
    if test.isOpened():
        ret, _ = test.read()
        if ret:
            cap = test
            print(f"Càmera oberta correctament a l'índex {idx}")
            break
        else:
            test.release()
    else:
        test.release()

if cap is None or not cap.isOpened():
    print("ERROR: No s'ha pogut obrir cap càmera.")
    print("Comprova que:")
    print("  1. La càmera està connectada.")
    print("  2. Cap altra aplicació l'està fent servir (Zoom, Teams, el joc SIGNO, etc.).")
    print("  3. Tens permisos per accedir-hi (Configuració de Windows > Privacitat > Càmera).")
    exit(1)

# NOMÉS NÚMEROS
gestos = ["0","1","2","3","4","5","6","7","8","9"]

index = 0
mostres_per_gest = 150
comptador = 0

csv_file = "dades_entrenament.csv"
file_exists = os.path.exists(csv_file)
f = open(csv_file, "a", newline="")
writer = csv.writer(f)
if not file_exists:
    writer.writerow([f"x{i}" for i in range(63)] + ["label"])

print(f"Gest actual: {gestos[index]}. Prem ESPAI per capturar, ESC per sortir.")
print("Assegura't que la finestra de la càmera estigui ACTIVA i clica-la si cal.")

fails_consecutius = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        fails_consecutius += 1
        if fails_consecutius > 30:
            print("ERROR: No es poden llegir frames de la càmera. Sortint...")
            break
        time.sleep(0.05)
        continue
    fails_consecutius = 0

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        pts = hand.landmark
        ox, oy, oz = pts[0].x, pts[0].y, pts[0].z
        escala = ((pts[9].x - ox) ** 2 + (pts[9].y - oy) ** 2) ** 0.5
        if escala == 0:
            escala = 1
        fila = []
        for p in pts:
            fila.extend([(p.x - ox) / escala, (p.y - oy) / escala, (p.z - oz) / escala])

        mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        cv2.putText(frame, f"Num: {gestos[index]} ({comptador}/{mostres_per_gest})",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "ESPAI = capturar | ESC = sortir",
                    (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            writer.writerow(fila + [gestos[index]])
            comptador += 1
            print(f"Capturat {comptador}/{mostres_per_gest} per a {gestos[index]}")
            if comptador >= mostres_per_gest:
                index += 1
                comptador = 0
                if index >= len(gestos):
                    print("Has acabat tots els números!")
                    break
                print(f"Següent número: {gestos[index]}")
        elif key == 27:  # ESC
            print("Sortint per ESC...")
            break
    else:
        cv2.putText(frame, f"Num: {gestos[index]} - Mostra la mà",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Recollida de dades - Números", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        print("Sortint per ESC...")
        break

f.close()
cap.release()
cv2.destroyAllWindows()
print("Recollida finalitzada.")