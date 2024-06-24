import cv2
from deepface import DeepFace

def detectar_emocoes(frame):
    try:
        resultado = DeepFace.analyze(frame, actions=['emotion'])
        emocoes = resultado[0]['emotion']
        emocao_dominante = max(emocoes, key=emocoes.get)
        
        traducao_emocoes = {
            'angry': 'Raiva',
            'disgust': 'Nojo',
            'fear': 'Medo',
            'happy': 'Feliz',
            'sad': 'Triste',
            'surprise': 'Surpresa',
            'neutral': 'Neutro'
        }
        emocao_dominante_pt = traducao_emocoes.get(emocao_dominante, emocao_dominante)
        
        face_rect = resultado[0]['region']
        x, y, w, h = face_rect['x'], face_rect['y'], face_rect['w'], face_rect['h']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emocao_dominante_pt, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    except Exception as e:
        print(f"Erro: {e}")
    
    return frame

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (640, 480))
    
    frame = detectar_emocoes(frame)
    
    cv2.imshow('Reconhecimento de Emoções', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
