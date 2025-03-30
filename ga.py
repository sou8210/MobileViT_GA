import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.quantization
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization.qconfig_mapping import QConfigMapping
import torch.ao.quantization.observer as ob
import copy
from model import mobilevit_xxs
from custom_model import MobileViT_xxs, CustomLinear, set_calib_true_for_all_custom_linear, set_calib_false_for_all_custom_linear
import eval


class MobileViTQuantizer:
    def __init__(self, pretrained_path, calibration_loader):
        self.original_model = mobilevit_xxs()
        self.original_model.load_state_dict(torch.load(pretrained_path))
        self.calibration_loader = calibration_loader

        self.qconfig_mapping = QConfigMapping().set_global(
            torch.quantization.QConfig(
                activation=ob.MinMaxObserver.with_args(dtype=torch.quint8),
                weight=ob.MinMaxObserver.with_args(dtype=torch.qint8)
            )
        )

    def calibrate(self, model):
        model.eval()
        with torch.no_grad():
            for images, _ in self.calibration_loader:
                model(images)

    def evaluate_accuracy(self, model):
        criterion = nn.CrossEntropyLoss()
        top1, _ = eval.evaluate(model, criterion, eval.test_loader)
        return top1.avg

    def quantize_model(self, bit_config):
        model = copy.deepcopy(self.original_model)
        bit8_model = torch.quantization.quantize_dynamic(copy.deepcopy(model), dtype=torch.qint8)
        bit4_model = MobileViT_xxs()
        bit4_model.load_state_dict(model.state_dict())

        # Apply quantization based on bit_config
        for idx, bit_choice in enumerate(bit_config[:10]):
            layer_idx = min(idx, len(model.mvit) - 1)
            transformer_layer = idx % len(model.mvit[layer_idx].transformer.layers)

            target_layer = model.mvit[layer_idx].transformer.layers[transformer_layer]
            if bit_choice == 0:  # 4bit
                model_layer = bit4_model.mvit[layer_idx].transformer.layers[transformer_layer]
            elif bit_choice == 1:  # 8bit
                model_layer = bit8_model.mvit[layer_idx].transformer.layers[transformer_layer]
            elif bit_choice == 2:  # 4bit activation
                model_layer = bit4_model.mvit[layer_idx].transformer.layers[transformer_layer]
                self.set_qact(model_layer)
            else:  # no quantization
                continue

            model.mvit[layer_idx].transformer.layers[transformer_layer] = model_layer

        if bit_config[9] in [0, 2]:  # fc layer quantization
            model.fc = bit4_model.fc
            if bit_config[9] == 2:
                self.set_qact(model.fc)
        elif bit_config[9] == 1:
            model.fc = bit8_model.fc

        # Quantize mv2 layers
        example_input = next(iter(self.calibration_loader))[0]
        for idx, bit_choice in enumerate(bit_config[10:]):
            if bit_choice == 0:
                prepared_mv2 = prepare_fx(model.mv2[idx], self.qconfig_mapping, example_input)
                self.calibrate(prepared_mv2)
                model.mv2[idx] = convert_fx(prepared_mv2)

        set_calib_true_for_all_custom_linear(model)
        self.calibrate(model)
        set_calib_false_for_all_custom_linear(model)

        return model

    @staticmethod
    def set_qact(layer):
        for module in layer.modules():
            if isinstance(module, CustomLinear):
                module.qact = True


# Genetic Algorithm related functions

class Individual:
    def __init__(self, genom, quantizer):
        self.genom = genom
        self.quantizer = quantizer
        self.fitness = self.evaluate()

    def evaluate(self):
        model = self.quantizer.quantize_model(self.genom)
        acc = self.quantizer.evaluate_accuracy(model)

        mvit_bits = sum([4 if g == 0 else 8 if g == 1 else 4 if g == 2 else 32 for g in self.genom[:10]])
        mv2_bits = sum([8 if g == 0 else 32 for g in self.genom[10:]])
        act_bits = sum([4 if g == 2 else 32 for g in self.genom[:10]])

        fitness = acc + 640/mvit_bits + 56/mv2_bits + 640/act_bits
        return fitness

    def mutate(self):
        idx = np.random.randint(len(self.genom))
        if idx < 10:
            self.genom[idx] = np.random.choice([0, 1, 2, 3])
        else:
            self.genom[idx] = 1 - self.genom[idx]
        self.fitness = self.evaluate()


def select(generation):
    weights = np.array([ind.fitness for ind in generation])
    probabilities = weights / weights.sum()
    return np.random.choice(generation, len(generation), p=probabilities)


def crossover(parent1, parent2):
    point = np.random.randint(1, len(parent1.genom)-1)
    child_genom = np.concatenate((parent1.genom[:point], parent2.genom[point:]))
    return child_genom


def ga_solve(quantizer, generations=10, pop_size=10):
    generation = [Individual(np.random.randint(0, 4, 10).tolist() + np.random.randint(0, 2, 7).tolist(), quantizer)
                  for _ in range(pop_size)]

    for gen in range(generations):
        print(f'Generation {gen}: Best fitness = {max(generation, key=lambda x: x.fitness).fitness:.2f}')
        selected = select(generation)
        next_gen = []
        for i in range(0, pop_size, 2):
            child_genom = crossover(selected[i], selected[i+1])
            child = Individual(child_genom, quantizer)
            if np.random.rand() < 0.1:
                child.mutate()
            next_gen.append(child)
        generation = next_gen

    best_individual = max(generation, key=lambda x: x.fitness)
    print('Best genom:', best_individual.genom)
    return best_individual


# Example usage
transform_calib = transforms.Compose([
    transforms.Resize(eval.target_size),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
calib_loader = DataLoader(eval.train_dataset, batch_size=128, shuffle=True)

quantizer = MobileViTQuantizer('cifar10_mobilevit_xxs_epoch75.pth', calib_loader)
best_solution = ga_solve(quantizer)
