import ast
import contextlib
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]

class Tensor:
    def __init__(self, values=()):
        self.values = list(values)
    @property
    def shape(self):
        return (len(self.values),)
    def __len__(self):
        return len(self.values)
    def __getitem__(self, key):
        if isinstance(key, tuple):
            key = key[0]
        result = self.values[key]
        return Tensor(result) if isinstance(key, slice) else result
    def cuda(self):
        return self
    def cpu(self):
        return self
    def to(self, device):
        return self

class Torch:
    Tensor = Tensor
    tensor = Tensor
    @staticmethod
    def cat(items, dim=0):
        return Tensor(v for item in items for v in item.values)
    no_grad = staticmethod(contextlib.nullcontext)

def function_from_source(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]

class EvaluationChecks(unittest.TestCase):
    def test_labels_and_cached_features(self):
        dataset = SimpleNamespace(attr2idx={'a': 0}, obj2idx={'o': 0},
                                  pairs=[('a', 'o')] * 5)
        batch = [Tensor([10]), Tensor([99]), Tensor([1]), Tensor([2]), Tensor([3])]
        config = SimpleNamespace(eval_batch_size=1, num_workers=0, text_encoder_batch_size=2)
        class Model:
            def eval(self): pass
            def __call__(self, data, pairs): return Tensor([7])
            def logit_infer(self, predict, pairs): return predict
            def loss_calu(self, predict, data): return SimpleNamespace(item=lambda: 0.)
            def encode_text_for_open(self, pairs):
                return [Tensor(range(len(pairs))), Tensor([11]), Tensor([22]), Tensor([33] * len(pairs))]
            def forward_for_open(self, data, features):
                assert [len(f) for f in features] == [5, 1, 1, 5]
                assert features[3].values == [33] * 5
                return Tensor([7])
        for path, name in [('test.py', 'predict_logits'),
                           ('test_MPPE.py', 'predict_logits'),
                           ('test_MPPE.py', 'predict_logits_text_first')]:
            env = dict(torch=Torch, DataLoader=lambda *a, **k: [batch],
                       tqdm=lambda iterable, **kw: iterable)
            result = function_from_source(path, name, env)(Model(), dataset, config)
            self.assertEqual([t.values for t in result[1:4]], [[1], [2], [3]])

    def test_cli_overrides_yaml(self):
        spec = importlib.util.spec_from_file_location('mppe_parameters', ROOT/'parameters.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = module.parser.parse_args([])
        config.open_world = False
        config.eval_batch_size = 30
        config.load_model = 'yaml.pt'
        config = module.parser.parse_args(['--open_world', 'true', '--load_model', 'cli.pt'], namespace=config)
        self.assertTrue(config.open_world)
        self.assertEqual(config.load_model, 'cli.pt')
        self.assertEqual(config.eval_batch_size, 30)
        self.assertFalse(module.parser.parse_args(['--open_world', 'False']).open_world)

    def test_pe_config_aliases(self):
        tree = ast.parse((ROOT/'model/MPPE.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MPPE')
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        loop = next(n for n in init.body if isinstance(n, ast.For))
        config = SimpleNamespace(pair_pe_a2o_loss_weight=1, pair_pe_o2a_loss_weight=0,
                                 pair_pe_a2o_inference_weight=1, pair_pe_o2a_inference_weight=0)
        exec(compile(ast.Module(body=[loop], type_ignores=[]), 'alias', 'exec'), {'config':config})
        self.assertEqual(config.pair_cam_a2o_loss_weight, 1)
        config.pair_cam_a2o_loss_weight = 2
        exec(compile(ast.Module(body=[loop], type_ignores=[]), 'alias', 'exec'), {'config':config})
        self.assertEqual(config.pair_cam_a2o_loss_weight, 2)

if __name__ == '__main__':
    unittest.main()
