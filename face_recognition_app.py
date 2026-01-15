#!/usr/bin/env python3
"""
Aplicativo simples de reconhecimento facial (coleta + treino + predição).

Fluxo:
1) O usuário envia quantas imagens quiser e informa o nome/pessoa a que cada imagem pertence.
2) O script recorta a(s) face(s) detectada(s) e armazena em `dataset/<nome>/`.
3) Quando o usuário digitar `done`, o script treina um reconhecedor (LBPH por padrão; opcional CNN se TensorFlow estiver disponível).
4) Por fim, o usuário fornece uma imagem para reconhecimento; o script identifica e salva `output.jpg` com anotações.

Requisitos: `opencv-contrib-python`, `numpy`. Opcional: `mtcnn` para detecção baseada em rede e `tensorflow` para classificação CNN.
"""

import os
import sys
import cv2
import numpy as np
import pickle
from pathlib import Path
try:
    from mtcnn import MTCNN
    HAS_MTCNN = True
except Exception:
    HAS_MTCNN = False
try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except Exception:
    HAS_TF = False


CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
LBPH_CONFIDENCE_THRESHOLD = 80.0
USE_TF_DETECTOR = os.environ.get('FACE_USE_TF_DETECTOR', '1' if HAS_MTCNN else '0').lower() in ('1', 'true', 'yes')
mtcnn_detector = None


def ensure_dirs():
    Path('dataset').mkdir(exist_ok=True)
    Path('trainer').mkdir(exist_ok=True)


def detect_faces(image, scaleFactor=1.3, minNeighbors=5):
    global mtcnn_detector
    if USE_TF_DETECTOR and HAS_MTCNN:
        if mtcnn_detector is None:
            mtcnn_detector = MTCNN()
        if len(image.shape) == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image
        results = mtcnn_detector.detect_faces(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        faces = []
        for r in results:
            x, y, w, h = r.get('box', [0, 0, 0, 0])
            x = max(0, int(x))
            y = max(0, int(y))
            w = max(1, int(w))
            h = max(1, int(h))
            faces.append((x, y, w, h))
        return faces
    else:
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detector = cv2.CascadeClassifier(CASCADE_PATH)
        faces = detector.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        return faces


def is_image_file(path):
    ext = str(path).lower()
    return ext.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))


def collect_images_interactive():
    print('\n--- Coleta de imagens para treinamento ---')
    print('Digite o caminho da foto ou da pasta com fotos, ou `done`.')
    print('Informe o nome da pessoa correspondente. Se houver várias faces, escolha um índice ou digite `all`.')

    while True:
        src_path = input('\nCaminho da foto/pasta (ou done): ').strip()
        if src_path.lower() in ('done', 'd'):
            break
        if not os.path.exists(src_path):
            print('Arquivo/pasta não encontrada, tente novamente.')
            continue

        if os.path.isdir(src_path):
            name = input('Nome da pessoa para todas as fotos desta pasta: ').strip()
            if not name:
                print('Nome inválido, tente novamente.')
                continue
            files = [str(p) for p in Path(src_path).glob('*') if is_image_file(str(p))]
            if not files:
                print('Nenhuma imagem encontrada na pasta.')
                continue
            for img_path in files:
                image = cv2.imread(img_path)
                if image is None:
                    continue
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = detect_faces(gray)
                if len(faces) == 0:
                    continue
                if len(faces) > 1:
                    print(f'{Path(img_path).name}: {len(faces)} faces detectadas.')
                    for i, (x, y, w, h) in enumerate(faces):
                        print(f'  [{i}] x={x} y={y} w={w} h={h}')
                    choice = input('Índice a salvar ou `all` para todas (enter para 0): ').strip().lower()
                    if choice in ('all', 'a'):
                        idxs = list(range(len(faces)))
                    elif choice == '':
                        idxs = [0]
                    else:
                        try:
                            idxs = [int(choice)]
                        except Exception:
                            idxs = [0]
                else:
                    idxs = [0]
                for idx in idxs:
                    x, y, w, h = faces[idx]
                    face_img = gray[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (200, 200))
                    person_dir = Path('dataset') / name
                    person_dir.mkdir(parents=True, exist_ok=True)
                    next_i = len(list(person_dir.glob('*.jpg')))
                    out_path = person_dir / f'{next_i:03d}.jpg'
                    cv2.imwrite(str(out_path), face_img)
                    print(f'Salvo {out_path}')
            continue

        name = input('Nome da pessoa para essa foto: ').strip()
        if not name:
            print('Nome inválido, tente novamente.')
            continue
        image = cv2.imread(src_path)
        if image is None:
            print('Não foi possível ler a imagem, pulei.')
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)
        if len(faces) == 0:
            print('Nenhuma face detectada nesta imagem. Pulei.')
            continue
        idxs = [0]
        if len(faces) > 1:
            print(f'Foram detectadas {len(faces)} faces nesta imagem.')
            for i, (x, y, w, h) in enumerate(faces):
                print(f'  [{i}] x={x} y={y} w={w} h={h}')
            choice = input('Índice a salvar ou `all` para todas (enter para 0): ').strip().lower()
            if choice in ('all', 'a'):
                idxs = list(range(len(faces)))
            elif choice == '':
                idxs = [0]
            else:
                try:
                    idxs = [int(choice)]
                except Exception:
                    idxs = [0]
        for idx in idxs:
            x, y, w, h = faces[idx]
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))
            person_dir = Path('dataset') / name
            person_dir.mkdir(parents=True, exist_ok=True)
            next_i = len(list(person_dir.glob('*.jpg')))
            out_path = person_dir / f'{next_i:03d}.jpg'
            cv2.imwrite(str(out_path), face_img)
            print(f'Salvo {out_path}')


def train_recognizer(method='lbph', progress_callback=None):
    if progress_callback: progress_callback('Iniciando treinamento...')
    dataset_dir = Path('dataset')
    if not dataset_dir.exists():
        print('Pasta dataset/ não encontrada. Colete imagens primeiro.')
        return False

    faces = []
    labels = []
    label_ids = {}
    current_id = 0

    for person_dir in sorted(dataset_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        if name not in label_ids:
            label_ids[name] = current_id
            current_id += 1
        id_ = label_ids[name]
        for img_path in person_dir.glob('*.jpg'):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(id_)

    if len(faces) == 0:
        print('Nenhuma imagem para treinar. Adicione fotos ao dataset/.')
        return False

    method = (method or 'lbph').strip().lower()
    if method == 'cnn':
        if not HAS_TF:
            print('TensorFlow não disponível. Instale `tensorflow` ou escolha LBPH.')
            return False
        X = np.stack([f for f in faces]).astype('float32') / 255.0
        X = np.expand_dims(X, -1)
        y = np.array(labels, dtype=np.int32)
        num_classes = len(set(labels))
        model = keras.Sequential([
            keras.layers.Input(shape=(200, 200, 1)),
            keras.layers.Conv2D(16, 3, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(32, 3, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(num_classes, activation='softmax'),
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        if progress_callback: progress_callback('Treinando CNN (isso pode demorar)...')
        model.fit(X, y, epochs=10, batch_size=16, verbose=1)
        model_path = Path('trainer') / 'model_tf.h5'
        model.save(str(model_path))
        with open('trainer/labels.pickle', 'wb') as f:
            pickle.dump(label_ids, f)
        print(f'Treino finalizado. Modelo salvo em {model_path} e labels em trainer/labels.pickle')
        return True
    else:
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
        except Exception:
            print('ERRO: cv2.face não disponível. Instale `opencv-contrib-python` e tente novamente.')
            return False
        recognizer.train(faces, np.array(labels))
        model_path = Path('trainer') / 'trainer.yml'
        recognizer.write(str(model_path))
        with open('trainer/labels.pickle', 'wb') as f:
            pickle.dump(label_ids, f)
        print(f'Treino finalizado. Modelo salvo em {model_path} e labels em trainer/labels.pickle')
        return True


def predict_on_image(input_image=None):
    labels_path = Path('trainer') / 'labels.pickle'
    model_tf_path = Path('trainer') / 'model_tf.h5'
    model_lbph_path = Path('trainer') / 'trainer.yml'
    if not labels_path.exists() or (not model_tf_path.exists() and not model_lbph_path.exists()):
        print('Modelo não encontrado. Treine primeiro usando a etapa de coleta e treino.')
        return None

    with open(labels_path, 'rb') as f:
        label_ids = pickle.load(f)
    id_labels = {v: k for k, v in label_ids.items()}

    use_tf = HAS_TF and model_tf_path.exists()
    if use_tf:
        model = keras.models.load_model(str(model_tf_path))
    else:
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
        except Exception:
            print('ERRO: cv2.face não disponível. Instale `opencv-contrib-python` ou treine com CNN.')
            return None
        recognizer.read(str(model_lbph_path))

    if input_image is None:
        img_path = input('\nCaminho da imagem para reconhecer: ').strip()
        if not os.path.exists(img_path):
            print('Arquivo não encontrado.')
            return None
        image = cv2.imread(img_path)
    else:
        image = input_image

    if image is None:
        print('Erro ao ler a imagem.')
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    if len(faces) == 0:
        print('Nenhuma face detectada na imagem de teste.')
        return
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        try:
            face_img = cv2.resize(face_img, (200, 200))
        except Exception:
            pass
        if use_tf:
            arr = (face_img.astype('float32') / 255.0)[None, ..., None]
            probs = model.predict(arr, verbose=0)[0]
            label_id = int(np.argmax(probs))
            confidence = float(np.max(probs) * 100.0)
            name = id_labels.get(label_id, 'desconhecido')
            text = f'{name} ({confidence:.1f}%)'
        else:
            label_id, confidence = recognizer.predict(face_img)
            name = id_labels.get(label_id, 'desconhecido')
            if confidence > LBPH_CONFIDENCE_THRESHOLD:
                name = 'desconhecido'
            text = f'{name} ({confidence:.1f})'
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    out_path = 'output.jpg'
    cv2.imwrite(out_path, image)
    print(f'Resultado salvo em {out_path}.')
    return image


def main():
    ensure_dirs()
    print('Aplicativo de reconhecimento facial — coleta, treino e predição')
    print('Certifique-se de ter instalado: opencv-contrib-python, numpy')
    if USE_TF_DETECTOR and HAS_MTCNN:
        print('Detecção: MTCNN')
    else:
        print('Detecção: Haar Cascade')
    collect_images_interactive()
    method_in = input('Método de classificação [lbph/cnn] (padrão lbph): ').strip().lower()
    if method_in not in ('lbph', 'cnn'):
        method_in = 'lbph'
    trained = train_recognizer(method_in)
    if trained:
        predict_on_image()


if __name__ == '__main__':
    main()