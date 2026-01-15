import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import os
from face_recognition_app import ensure_dirs, detect_faces, train_recognizer, predict_on_image, HAS_TF

# Configuração da página
st.set_page_config(page_title="Reconhecimento Facial", layout="wide")
ensure_dirs()

st.title("Sistema de Reconhecimento Facial")

# Sidebar para navegação
menu = st.sidebar.selectbox("Menu", ["Coleta de Dados", "Treinamento", "Reconhecimento"])

if menu == "Coleta de Dados":
    st.header("1. Coleta de Imagens para o Dataset")
    
    person_name = st.text_input("Nome da Pessoa:")
    uploaded_files = st.file_uploader("Escolha imagens...", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])
    
    if st.button("Processar e Salvar Faces"):
        if not person_name:
            st.error("Por favor, informe o nome da pessoa.")
        elif not uploaded_files:
            st.error("Por favor, selecione pelo menos uma imagem.")
        else:
            progress_bar = st.progress(0)
            person_dir = Path('dataset') / person_name
            person_dir.mkdir(parents=True, exist_ok=True)
            
            saved_count = 0
            for i, uploaded_file in enumerate(uploaded_files):
                # Converter arquivo para imagem OpenCV
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if image is not None:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = detect_faces(gray)
                    
                    for (x, y, w, h) in faces:
                        face_img = gray[y:y+h, x:x+w]
                        face_img = cv2.resize(face_img, (200, 200))
                        
                        next_i = len(list(person_dir.glob('*.jpg')))
                        out_path = person_dir / f'{next_i:03d}.jpg'
                        cv2.imwrite(str(out_path), face_img)
                        saved_count += 1
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.success(f"Processamento concluído! {saved_count} faces salvas para '{person_name}'.")

elif menu == "Treinamento":
    st.header("2. Treinar o Modelo")
    
    dataset_path = Path('dataset')
    if not dataset_path.exists() or not any(dataset_path.iterdir()):
        st.warning("O dataset está vazio. Vá para 'Coleta de Dados' primeiro.")
    else:
        st.info("Pessoas encontradas no dataset:")
        names = [d.name for d in dataset_path.iterdir() if d.is_dir()]
        st.write(", ".join(names))
        
        method = st.radio("Método de Treinamento:", ["LBPH", "CNN"] if HAS_TF else ["LBPH"])
        
        if st.button("Iniciar Treinamento"):
            status_text = st.empty()
            
            def update_status(msg):
                status_text.text(msg)
            
            success = train_recognizer(method=method.lower(), progress_callback=update_status)
            
            if success:
                st.success("Modelo treinado com sucesso!")
            else:
                st.error("Erro durante o treinamento.")

elif menu == "Reconhecimento":
    st.header("3. Reconhecimento Facial em Imagem")
    
    labels_path = Path('trainer') / 'labels.pickle'
    if not labels_path.exists():
        st.warning("Nenhum modelo treinado encontrado. Vá para 'Treinamento' primeiro.")
    else:
        test_file = st.file_uploader("Escolha uma imagem para reconhecimento...", type=['jpg', 'jpeg', 'png'])
        
        if test_file is not None:
            # Converter para OpenCV
            file_bytes = np.asarray(bytearray(test_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if st.button("Executar Reconhecimento"):
                result_image = predict_on_image(input_image=image)
                
                if result_image is not None:
                    # Converter BGR para RGB para o Streamlit
                    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    st.image(result_rgb, caption="Resultado do Reconhecimento", use_container_width=True)
                else:
                    st.error("Não foi possível processar a imagem ou nenhuma face foi detectada.")
