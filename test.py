from utils.dataset import Dataset
from utils.model import MLP
import os
import re

save_path = 'hidden_2048_2048_relu_lr_0.1_decay_0.95_l2_0.0001'

print(save_path)

def parse_string(input_string):
    pattern = r'hidden_(\d+)_(\d+)_(\w+)_lr_([\d.]+)_decay_([\d.]+)_l2_([\d.]+)'
    match = re.match(pattern, input_string)
    if match:
        dim_hidden = [int(match.group(1)), int(match.group(2))]
        activation_function = match.group(3)
        learning_rate = float(match.group(4))
        lr_decay = float(match.group(5))
        l2_reg = float(match.group(6))
        return dim_hidden, activation_function, learning_rate, lr_decay, l2_reg
    else:
        return None


result = parse_string(save_path)
if result:
    dim_hidden, activation_function, learning_rate, lr_decay, l2_reg = result
    print(f"dim_hidden: {dim_hidden}")
    print(f"activation_function: {activation_function}")
else:
    print("解析失败，请检查输入字符串格式。")


test_dataset = Dataset(type='test')
test_X, test_Y = test_dataset.get_all_data()

save_path=os.path.join('./saves',save_path)

model = MLP(input_size=3072, dim_hidden=dim_hidden, output_size=10, activation_function=activation_function)
model.load_from_trained(os.path.join(save_path, 'model.pkl'))
model.forward(test_X)
acc = model.accuracy(test_Y)
print(f"test accuracy: {acc}")