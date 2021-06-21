from keras.applications import vgg16

IMG_ROWS, IMG_COLS = 224, 224


models = vgg16.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_ROWS, IMG_COLS, 3)
)

print("Deu certo")
