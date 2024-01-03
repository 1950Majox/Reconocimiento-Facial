import cv2
import streamlit as st
from simple_facerec import SimpleFacerec
import os 
from PIL import Image
import threading


ruta_imagenes = "imagenes"

if not os.path.exists(ruta_imagenes):
    os.makedirs(ruta_imagenes)


ruta_logo = "logo2.png"

# Mostrar el logo
logo = Image.open(ruta_logo)
st.sidebar.image(logo, use_column_width=True)


def pantalla1():
    st.title("Reconocimiento Facial ")

    sfr = SimpleFacerec()
    sfr.load_encoding_images("imagenes/")

    cap = cv2.VideoCapture(0)

    # Agregamos un elemento de Streamlit para mostrar el video
    video_element = st.empty()

    while True:
        ret, frame = cap.read()

        face_locations, face_names = sfr.detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        # Actualizamos el contenido del elemento de video en Streamlit
        video_element.image(frame, channels="BGR", use_column_width=True)

    cap.release()

def pantalla2():
    st.title("Cargar y Guardar Imágenes en Streamlit")

    # Subir imagen desde el usuario
    imagen_subida = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

    # Guardar la imagen en la carpeta "imagenes"
    if imagen_subida is not None:
        # Mostrar la imagen
        st.image(imagen_subida, caption="Imagen subida", use_column_width=True)

        # Guardar la imagen en la carpeta
        imagen_path = os.path.join(ruta_imagenes, imagen_subida.name)
        with open(imagen_path, "wb") as f:
            f.write(imagen_subida.read())

        st.success(f"La imagen se ha guardado en {imagen_path}")



def main():
    st.sidebar.title("NAVEGACIÓN")
    seleccion = st.sidebar.radio("Ir a:", ("RECONOCIMIENTO FACIAL", "GUARDAR NUEVAS IMAGENES"))

    if seleccion == "RECONOCIMIENTO FACIAL":
        pantalla1()
    elif seleccion == "GUARDAR NUEVAS IMAGENES":
        pantalla2()

if __name__ == "__main__":
    main()
