import streamlit as st
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
#from PIL import Image

st.set_page_config(page_title='reFace - Ilhom Jabborov')
t1, t2 = st.columns((0.25,1))
t1.image('./icon.jpg',width=120)
t2.title("reFace - Sun'iy Zako")
t2.markdown("**Muallif:** <span style='color: green;'>Ilhom Jabborov</span> **| Loyiha:** <span style='color: green;'>reFace AI</span> **| Yil:** <span style='color: green;'>2024</span>", unsafe_allow_html=True)


tab1, tab2, tab3 ,tab4= st.tabs(('Bosh Sahifa',"Qo'llanma","Sinab ko'rish",'Natija'))


app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

path_model = './inswapper_128.onnx'
swapper = insightface.model_zoo.get_model(path_model, download=False, download_zip=False)

with tab1 :
    st.image("./faceY.jpg")
    st.markdown(""" <html>
<head>
  <meta charset="UTF-8">
  <style>
    /* CSS styles go here */

    h2 {
      color: #FFFC33;
      text-align: center;
    }

    p {
      color: #FFFFFF;
      line-height: 1.6;
    }
  </style>
</head>
<body>
  <div class="container">
    <h2>reFace AI-ga xush kelibsiz !</h2>
    <p>reFace AI - qiziqarli va innovatsion tasvirni manipulyatsiya qilish uchun sizning xizmatingizda. Bizning veb-saytimiz foydalanuvchilarga qiziqarli va interaktiv tajribani taqdim etish uchun sun'iy intellektning kuchidan foydalanadi.</p>
    <hr>  
    <p>Innovatsiyalar va sunʼiy intellekt yordamida tasvirni manipulyatsiya qilishning cheksiz imkoniyatlarini oʻrganishda davom etayotganimizda bizga qoʻshiling. Keling, texnologiya bilan o'zaro munosabatimizni birgalikda qayta tasavvur qilaylik.</p>  
    <hr>
    <p>Ijtimoiy tarmoqdagi postlaringizga hazil qo‘shmoqchimisiz yoki sun’iy intellekt imkoniyatlariga qiziqasizmi, reFace AI-da hamma uchun nimadir bor.</p><br>
    <h2>Xo'sh, nimani kutmoqdasiz ? </h2>
    <p>Suratingizni yuklang va zavqlanishni boshlang !</p>       
    </div>
</body>
</html>
                
                 """, unsafe_allow_html=True)
    
with tab2 :
    st.title('Saytdan Qanday Foydalanaman ?')
    with st.expander("Matnli Qo'llanma"):
        st.markdown("""
                    ##### Bu qanday ishlaydi ? :
                    * Bu oddiy! Foydalanuvchilar o'zlarining fotosuratlarini yuklaydilar va keyin yuzlarini almashtirmoqchi bo'lgan boshqa rasmni tanlaydilar. Bizning ilg‘or AI algoritmlarimiz qolgan ishlarni o'zi amalga oshiradi : kulguli yoki hayratlanarli natijalarni yaratish uchun yuzlarni muammosiz aralashtiradi.
                    * Ishlov berilgan rasmni **"Natija"** bo'limida ko'rishingiz mumkin
                    """)
        
        st.divider()
        
        audio_file = open("./music.mp3", "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mpeg")
    
    with st.expander("Video Qo'llanma"):
        st.subheader('Saytdan foydalanish')
        video_file = open('./video.webm', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

        st.divider()

with tab3 :

    my_face = st.file_uploader("Rasmingizni Kiriting", type=["png", "jpg", "jpeg"])
    reFace = st.file_uploader("O'zgartiriladigan Rasmni Kiriting", type=["png", "jpg", "jpeg"])

    if my_face is not None:
        # Read the file content and convert it to numpy array
        my_face_content = np.asarray(bytearray(my_face.read()), dtype=np.uint8)
        rob = cv2.imdecode(my_face_content, cv2.IMREAD_COLOR)

        st.image(rob, caption='Siz yuklagan rasm', channels='BGR')


with tab4 :
    if reFace is not None:
        path_img = reFace
        full_img_content = np.asarray(bytearray(reFace.read()), dtype=np.uint8)
        full_img = cv2.imdecode(full_img_content, cv2.IMREAD_COLOR)

        rob_faces = app.get(rob)
        rob_face = rob_faces[0]

        faces = app.get(full_img)
        res = full_img.copy()

        for face in faces:
            res = swapper.get(res, face, rob_face, paste_back=True)

        # res = cv2.resize(res,(400,400))
        st.image(res, caption='Ishlov berilgan rasm', channels='BGR')



    
