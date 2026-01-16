# Sistema de Reconhecimento Facial

Aplicativo didático para coletar imagens rotuladas, treinar um reconhecedor e identificar rostos em uma imagem final. O fluxo permite adicionar quantas fotos e pessoas quiser, inclusive processando pastas inteiras e salvando todas as faces detectadas.

**Acesse a versão online (Gradio) hospedada no Hugging Face:** [Facial Recognition Space](https://huggingface.co/spaces/PedroM2626/facial-recognition)

## Conteúdo
- `app.py`: Interface web moderna usando Streamlit.
- `face_recognition_app.py`: script interativo (CLI) para coleta, treino e predição.
- `face_recognition_app.ipynb`: notebook com upload de imagens e visualização inline.
- `dataset/`: faces recortadas por pessoa em `dataset/<nome>/NNN.jpg`.
- `trainer/`: modelos e labels (`trainer.yml` para LBPH, `model_tf.h5` para CNN, `labels.pickle`).
- `requirements.txt`: dependências base e de notebook.

## Requisitos
- Python 3.9+ (Windows, macOS ou Linux)
- Dependências mínimas: `opencv-contrib-python`, `numpy`
- Opcionais:
  - Detecção por rede: `mtcnn`
  - Classificação por CNN: `tensorflow`

### Instalação (Windows PowerShell)
```
python -m pip install --upgrade pip
pip install opencv-contrib-python numpy
# opcionais
pip install mtcnn
pip install tensorflow
pip install jupyterlab ipywidgets matplotlib
```

## Uso (Interface Web - Recomendado)
1) Execute o Streamlit
```
streamlit run app.py
```
2) Navegue pelo menu lateral para coletar dados, treinar o modelo e realizar o reconhecimento.

## Uso (CLI)
1) Execute o aplicativo
```
python face_recognition_app.py
```

2) Coleta de imagens
- Informe caminho de uma foto ou de uma pasta com fotos.
- Dê o nome da pessoa correspondente.
- Se houver várias faces em uma imagem, escolha um índice ou digite `all` para salvar todas.
- Repita o processo quantas vezes quiser; finalize com `done`.

3) Treinamento
- Escolha o método: `lbph` (padrão) ou `cnn` (se TensorFlow estiver instalado).
- Saídas:
  - LBPH: `trainer/trainer.yml`
  - CNN (simples): `trainer/model_tf.h5`
  - Labels: `trainer/labels.pickle`

4) Predição
- Informe o caminho da imagem a reconhecer.
- Todas as faces detectadas serão rotuladas e salvas em `output.jpg`.

## Detecção de faces
- Padrão: Haar Cascade (`haarcascade_frontalface_default.xml`).
- Opcional: MTCNN (se `mtcnn` estiver instalado). Controle pelo ambiente:
  - PowerShell: `$env:FACE_USE_TF_DETECTOR=1` habilita MTCNN; `0` desativa.
  - Por padrão, se `mtcnn` estiver instalado, MTCNN é usado.

## Classificação
- LBPH: `cv2.face.LBPHFaceRecognizer_create()` (requer `opencv-contrib-python`).
- CNN: rede simples treinada do zero (`200x200x1`), salva em `trainer/model_tf.h5`.
- Limiar de desconhecido para LBPH: `LBPH_CONFIDENCE_THRESHOLD = 80.0`.

## Anexar imagens (upload) em Jupyter
Em terminal, “anexar” arquivos não é suportado; é necessário informar caminhos. Em Jupyter Notebook é possível fazer upload com `ipywidgets`.

### Passos
```
pip install jupyterlab ipywidgets opencv-contrib-python numpy mtcnn tensorflow
jupyter lab
```

Crie um notebook e execute a célula abaixo para enviar imagens e salvá-las em `dataset/<nome>/`:

```python
from ipywidgets import FileUpload, Text, Button, VBox
from pathlib import Path
import numpy as np, cv2
from face_recognition_app import ensure_dirs, detect_faces

ensure_dirs()
u = FileUpload(accept='.jpg,.jpeg,.png', multiple=True)
name = Text(description='Nome:')
btn = Button(description='Salvar faces')

def on_click(_):
    if not name.value:
        print('Informe um nome.'); return
    person_dir = Path('dataset')/name.value
    person_dir.mkdir(parents=True, exist_ok=True)
    for item in u.value.values():
        data = np.frombuffer(item['content'], dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)
        if not faces: continue
        for (x,y,w,h) in faces:
            face = cv2.resize(gray[y:y+h, x:x+w], (200,200))
            next_i = len(list(person_dir.glob('*.jpg')))
            cv2.imwrite(str(person_dir/f'{next_i:03d}.jpg'), face)
    print('Faces salvas em', person_dir)

btn.on_click(on_click)
VBox([name, u, btn])
```

Depois de coletar, volte ao terminal ou crie células adicionais para treinar e prever usando o script.

Além disso, o repositório já contém `face_recognition_app.ipynb` com células prontas para upload, treino e predição. Se desejar parear com o código usando Jupytext:

```
jupytext --sync face_recognition_app.ipynb
jupytext --set-formats ipynb,py:percent face_recognition_app.ipynb
```

## Perguntas frequentes
- “cv2.face não disponível”: instale `opencv-contrib-python`.
- Ambiente sem janelas: `output.jpg` é gerado mesmo sem `cv2.imshow`.
- Desempenho/qualidade: colete muitas imagens por pessoa, com variação de iluminação e ângulos.

## Estrutura do Projeto
```
.
├─ face_recognition_app.py
├─ face_recognition_app.ipynb
├─ README.md
├─ LICENSE.txt
├─ requirements.txt
├─ dataset/            # criado pelo script
└─ trainer/            # criado pelo script
```

## Transfer Learning
- O projeto não utiliza transfer learning com TensorFlow atualmente.
- O modo `cnn` treina uma rede simples do zero. Caso deseje TL (ex.: `MobileNetV2` congelado + cabeça densa), pode ser integrado mantendo a interface atual.

## Licença
- Consulte o arquivo `LICENSE`.