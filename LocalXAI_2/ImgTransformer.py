from PIL import Image

from torchvision import transforms


class ImgTransformer:
    transform = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
    ])
    transform_crop = transforms.Compose([
            transforms.Resize([373,373]),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
    ])
    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    def get_transformed_img(self, img_path, normalize=True, crop=True):
        img = Image.open(img_path)
        converted_img = img.convert('RGB')
        if crop:
            transformed_img = self.transform_crop(converted_img)
        else:
            transformed_img = self.transform(converted_img)
        if normalize:
            input_ = self.transform_normalize(transformed_img).unsqueeze(0)
        else:
            input_ = transformed_img.unsqueeze(0)
        return img, converted_img, transformed_img, input_
