import cv2
from deepface import DeepFace

# Função para detectar emoções e desenhar o retângulo com o nome da emoção
def detectar_emocoes(frame):
    try:
        resultado = DeepFace.analyze(frame, actions=['emotion'])
        emocoes = resultado[0]['emotion']
        emocao_dominante = max(emocoes, key=emocoes.get)
        
        # Traduzir a emoção dominante para português
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
        
        # Desenhar um retângulo ao redor do rosto e exibir a emoção detectada
        face_rect = resultado[0]['region']
        x, y, w, h = face_rect['x'], face_rect['y'], face_rect['w'], face_rect['h']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emocao_dominante_pt, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    except Exception as e:
        print(f"Erro: {e}")
    
    return frame

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Redimensionar o frame para reduzir o tempo de processamento
    frame = cv2.resize(frame, (640, 480))
    
    # Detectar emoções no frame
    frame = detectar_emocoes(frame)
    
    # Exibir o frame com a emoção detectada
    cv2.imshow('Reconhecimento de Emoções', frame)
    
    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura de vídeo e fechar as janelas
cap.release()
cv2.destroyAllWindows()