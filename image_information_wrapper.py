import cv2

from process_image import process_image, process_image_verbose, nicely_overlay_images


class ImageInformationWrapper:
    def __init__(
        self,
        name: str,
        input_directory: str,
        output_directory: str,
        input_subdirectory: str = "min/",
        also_create_nice_overlay_output: bool = False,
        overlay_output_directory: str = None,
    ):
        self.name = name
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.input_subdirectory = input_subdirectory
        self.also_create_nice_overlay_output = also_create_nice_overlay_output
        self.overlay_output_directory = overlay_output_directory

        self.loaded = False
        self.processed = False

    def get_input_path(self):
        return self.input_directory + self.input_subdirectory + self.name

    def get_output_path(self):
        return self.output_directory + self.name

    def get_overlay_path(self):
        return self.overlay_output_directory + self.name

    def load_img(self):
        self.img = cv2.imread(self.get_input_path(), cv2.IMREAD_GRAYSCALE,)
        self.loaded = True
        return self

    def process_img(self):
        if not self.loaded:
            raise ValueError("Image must be processed to save.")

        self.processed_img = process_image(self.img)
        self.processed = True
        return self

    def process_img_verbose(self):
        if not self.loaded:
            raise ValueError("Image must be processed to save.")

        self.processed_img = process_image_verbose(self.img)
        self.processed = True
        return self

    def save_img(self):
        if not self.processed:
            raise ValueError("Image must be processed to save.")
        cv2.imwrite(self.get_output_path(), self.processed_img)

        if self.also_create_nice_overlay_output:
            overlay = nicely_overlay_images(self.img, self.processed_img)
            cv2.imwrite(self.get_overlay_path(), overlay)

        return self
