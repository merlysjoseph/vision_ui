def detect_face(gray_image):
    faces = face_cascade.detectMultiScale(gray_image,scaleFactor=1.2,minNeighbors=10,minSize=(face_size, face_size))

    face_imgs = np.empty((len(faces), face_size, face_size, 3))
    for i, face in enumerate(faces):
        face_img, cropped = crop_face(frame, face, margin=40, size=face_size)
        (x, y, w, h) = cropped
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_imgs[i,:,:,:] = face_img

        if len(face_imgs) > 0:
        # predict ages and genders of the detected faces
            results = model.predict(face_imgs)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()
            label = "age={}, gender={}".format(int(predicted_ages[i]),
                                                "F" if predicted_genders[i][0] > 0.5 else "M")
    return (faces,label)

print("Buggy")