import argparse
import onnx_export
from models.common import *
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov5s.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    print(opt)

    f = opt.weights.replace('.pt', '.onnx')  
    img = torch.zeros((opt.batch_size, 3, *opt.img_size))  

    google_utils.attempt_download(opt.weights)
    model = torch.load(opt.weights, map_location=torch.device('cpu'))['model'].float()
    model.eval()
    model.fuse()

    model.model[-1].export = True 
    _ = model(img)  
    torch.onnx.export(model, img, f, verbose=False, opset_version=11, input_names=['images'],
                      output_names=['output'])  

    model = onnx_export.load(f) 
    onnx_export.checker.check_model(model) 
    print(onnx_export.helper.printable_graph(model.graph)) 
    print('Export complete' % f)