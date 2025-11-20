import onnx
import onnxruntime as ort
import numpy as np
import onnxsim
import json
import os
import re
from collections import OrderedDict
import copy
import argparse
import shutil
from glob import glob
from scipy.spatial.distance import cosine
import subprocess
import struct
from onnx import numpy_helper
try:
    # Relative import: suitable for in-package calls (recommended)
    from .infer_onnx_with_fp8 import (
        infer_onnx_with_custom_quant, is_cunstom_fp8_model, get_shape, get_model_quant_attrs)
    from .remove_duplicate_initializer import remove_duplicate_initializer
    from .remove_initializer_from_input import remove_initializer_from_input
except ImportError:
    from infer_onnx_with_fp8 import (
        infer_onnx_with_custom_quant, is_cunstom_fp8_model, get_shape, get_model_quant_attrs)
    from remove_duplicate_initializer import remove_duplicate_initializer
    from remove_initializer_from_input import remove_initializer_from_input

MAX_DEPTH = 32
cur_path = os.path.abspath(os.path.dirname(__file__))

def replace_special_charactor(tensor_name):
    out_tensor_name = tensor_name.replace("2F", "/")
    out_tensor_name = out_tensor_name.replace("2E", ".")
    out_tensor_name = out_tensor_name.replace("3A", ":")
    return out_tensor_name

def replace_special_charactor_inverse(tensor_name):
    out_tensor_name = tensor_name.replace("/", "2F")
    out_tensor_name = out_tensor_name.replace(".", "2E")
    out_tensor_name = out_tensor_name.replace(":", "3A")
    return out_tensor_name

def parse_model(model_):
    initi_names = [i.name for i in model_.graph.initializer]

    new_node_count = 0
    for node in model_.graph.node:
        if node.name == '':
            node.name = f"{node.op_type}_{new_node_count}"
            new_node_count += 1

    name_to_nodes = dict()
    itensor_to_nodes = dict()
    node_i_tnsrs = dict()
    node_o_tnsrs = dict()
    for node in model_.graph.node:
        name_to_nodes[node.name] = node
        node_i_tnsrs[node.name] = []
        node_o_tnsrs[node.name] = [replace_special_charactor_inverse(j) for j in node.output]
        for i_idx, i_ in enumerate(node.input):
            if i_ in initi_names:
                continue
            if i_ == '':
                continue
            norm_i_ = replace_special_charactor_inverse(i_)
            node_i_tnsrs[node.name].append(norm_i_)
            if norm_i_ not in itensor_to_nodes:
                itensor_to_nodes[norm_i_] = []
            if node.name not in itensor_to_nodes[norm_i_]:
                itensor_to_nodes[norm_i_].append(node.name)

    otensor_to_nodes = dict()
    for node in model_.graph.node:
        for i_idx,i_ in enumerate(node.output):
            if i_ == '':
                continue
            norm_i_ = replace_special_charactor_inverse(i_)
            if norm_i_ not in otensor_to_nodes:
                otensor_to_nodes[norm_i_] = []
            if node.name not in otensor_to_nodes[norm_i_]:
                otensor_to_nodes[norm_i_].append(node.name)

    graph_dict = OrderedDict()
    for node in model_.graph.node:
        graph_dict[node.name] = []
        for o in node.output:
            norm_o = replace_special_charactor_inverse(o)
            if norm_o in itensor_to_nodes:
                d_node_names = itensor_to_nodes[norm_o]
                for d_node_name in d_node_names:
                    if d_node_name not in graph_dict[node.name]:
                        graph_dict[node.name].append(d_node_name)

    inverse_graph_dict = OrderedDict()
    for node in model_.graph.node:
        inverse_graph_dict[node.name] = []
        for o in node.input:
            if o  in initi_names:
                continue
            norm_o = replace_special_charactor_inverse(o)
            if norm_o in otensor_to_nodes:
                d_node_names = otensor_to_nodes[norm_o]
                for d_node_name in d_node_names:
                    if d_node_name not in inverse_graph_dict[node.name]:
                        inverse_graph_dict[node.name].append(d_node_name)

    return {
        'name_to_nodes':name_to_nodes,
        'itensor_to_nodes':itensor_to_nodes,
        'otensor_to_nodes': otensor_to_nodes,
        'node_i_tnsrs': node_i_tnsrs,
        'node_o_tnsrs': node_o_tnsrs,
        'graph_dict':graph_dict,
        'inverse_graph_dict': inverse_graph_dict
        }

def can_reachable_dfs(graph, start_node, target_node):
    visited = set()

    def dfs(current):
        if current == target_node:
            return True

        visited.add(current)

        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
        return False

    return dfs(start_node)

def bfs_with_target(graph, start_node, target_node):
    from collections import deque

    all_paths = []
    queue = deque([(start_node, [start_node])])

    while queue:
        node, current_path = queue.popleft()

        if node == target_node:
            all_paths.append(current_path)
            continue

        for neighbor in graph.get(node, []):
            new_path = current_path + [neighbor]
            queue.append((neighbor, new_path))

    return all_paths

def dfs_with_target(graph, start_node, target_node):
    all_paths = []

    def dfs(current, path):
        current_length = len(path) + 1
        if current_length > MAX_DEPTH:
            return
        new_path = path + [current]

        if current == target_node:
            all_paths.append(new_path)
            return

        for neighbor in graph.get(current, []):
            if neighbor not in new_path:
                dfs(neighbor, new_path)

    dfs(start_node, [])
    return all_paths

def is_same_father(graph_dict, node_a, node_b):
    flag = False
    if node_a != node_b:
        for node_name in graph_dict:
            if node_a in graph_dict[node_name] and node_b in graph_dict[node_name] and \
                node_a not in graph_dict[node_b] and node_b not in graph_dict[node_a]:
                flag = True
    return flag

def merge_path(graph, paths, exit_paths):
    torp_sort_node_names = list(graph.keys())
    exit_path_strs = '_'.join(['_'.join(i) for i in exit_paths])
    paths_ = []
    for path_ in paths:
        if '_'.join(path_) in exit_path_strs:
            continue
        exit_paths.append(path_)
        paths_ += path_
    new_path = [i for i in torp_sort_node_names if i in paths_]
    return new_path


def org_model_to_tlf_model(org_model_info, modified_model_info):
    node_maps = OrderedDict()

    modified_model_paths = []
    org_model_paths = []

    org_model_node_names = list(org_model_info['name_to_nodes'].keys())

    # PATCH: 过滤掉 ArtFusedCallback 这类纯 NPU helper 节点，不要求它们和原始模型一一映射
    modified_model_node_names = [
        name for name, node in modified_model_info['name_to_nodes'].items()
        if node.op_type != 'ArtFusedCallback'
    ]

    org_model_node_idx = 0
    modified_model_node_idx = 0

    org_graph_dict = org_model_info['graph_dict']
    modified_graph_dict = modified_model_info['graph_dict']

    modified_model_node_nums = len(modified_model_node_names)
    while (len(org_model_node_names) and len(modified_model_node_names)):
        print('len(org_model_node_names):', len(org_model_node_names))
        print('len(modified_model_node_names):', len(modified_model_node_names))
        visited_modi_nodes = sum(modified_model_paths, []) \
            if len(modified_model_paths) > 0 else modified_model_paths
        visited_org_nodes = sum(org_model_paths, []) \
            if len(org_model_paths) > 0 else org_model_paths

        node_name = org_model_node_names[org_model_node_idx]
        modi_start_nodes = []
        modi_end_nodes = []
        tlf_nodes = []
        onnx_nodes = []
        find_org_nodes = []
        proposal_del_modified_model_node_names = []
        org_otnsr_not_in_modi_model = False
        modi_otnsr_not_in_org_model = False
        for i_tnsr in org_model_info['node_i_tnsrs'][node_name]:
            # some node maybe delete
            if i_tnsr not in modified_model_info['itensor_to_nodes']:
                continue
            for i in modified_model_info['itensor_to_nodes'][i_tnsr]:
                if i not in modi_start_nodes and i not in visited_modi_nodes \
                   and i in modified_model_node_names:
                    modi_start_nodes.append(i)
        for o_tnsr in org_model_info['node_o_tnsrs'][node_name]:
            # some node maybe merged
            if o_tnsr not in modified_model_info['otensor_to_nodes']:
                org_otnsr_not_in_modi_model = True
                continue
            for i in modified_model_info['otensor_to_nodes'][o_tnsr]:
                if i not in modi_end_nodes:
                    modi_end_nodes.append(i)

        if org_otnsr_not_in_modi_model:
            # clear modi_model found
            modi_start_nodes = []

            org_start_nodes = []
            org_end_nodes = []
            modified_model_node_idx = 0
            while (len(org_end_nodes) == 0):
                modi_search_count = 0
                while (len(org_start_nodes) == 0 and
                       modi_search_count < len(modified_model_node_names)):
                    modi_node_name = modified_model_node_names[modi_search_count]

                    for i_tnsr in modified_model_info['node_i_tnsrs'][modi_node_name]:
                        # some node maybe delete
                        if i_tnsr not in org_model_info['itensor_to_nodes']:
                            continue
                        for i in org_model_info['itensor_to_nodes'][i_tnsr]:
                            if i not in org_start_nodes and i not in visited_org_nodes \
                               and i in org_model_node_names:
                                org_start_nodes.append(i)
                    modi_search_count += 1

                for o_tnsr in modified_model_info['node_o_tnsrs'][modi_node_name]:
                    # some node maybe delete
                    if o_tnsr not in org_model_info['otensor_to_nodes']:
                        continue
                    for i in org_model_info['otensor_to_nodes'][o_tnsr]:
                        if i not in org_end_nodes:
                            org_end_nodes.append(i)

                modified_model_node_idx += (modi_search_count - 1)
                modified_model_node_idx += 1
                if modi_node_name not in proposal_del_modified_model_node_names:
                    proposal_del_modified_model_node_names.append(modi_node_name)
                else:
                    modi_node_name = modified_model_node_names[modified_model_node_idx]

                if len(proposal_del_modified_model_node_names) > modified_model_node_nums:
                    raise Exception('{} search path over limit'.format(modi_node_name))

            if len(org_end_nodes) == 0:
                raise Exception('{} search path error'.format(modi_node_name))

            if org_start_nodes == org_end_nodes:
                find_org_nodes = [org_start_nodes]
            else:
                for org_start_node in org_start_nodes:
                    for org_end_node in org_end_nodes:
                        find_org_nodes += dfs_with_target(
                            org_graph_dict, org_start_node, org_end_node)
        else:
            if modi_start_nodes == modi_end_nodes:
                tlf_nodes = [modi_start_nodes]
            else:
                for modi_start_node in modi_start_nodes:
                    for modi_end_node in modi_end_nodes:
                        tlf_nodes += dfs_with_target(
                            modified_graph_dict, modi_start_node, modi_end_node)

        assert (len(tlf_nodes) and len(find_org_nodes) == 0) or (
            len(find_org_nodes) and len(tlf_nodes) == 0
        ), 'tlf_nodes or find_org_nodes only one is needed'

        if len(tlf_nodes) and len(find_org_nodes) == 0:
            if len(tlf_nodes) == 1:
                modified_model_paths += tlf_nodes
                node_maps[node_name] = tlf_nodes[0]
            else:
                node_maps[node_name] = merge_path(
                    modified_graph_dict, tlf_nodes, modified_model_paths)
            for i in node_maps[node_name]:
                if i not in modified_model_node_names:
                    continue
                modified_model_node_names.remove(i)
            org_model_node_names.remove(node_name)

        if len(find_org_nodes) and len(tlf_nodes) == 0:
            org_node_tp = None
            if len(find_org_nodes) == 1:
                org_model_paths += find_org_nodes
                org_node_tp = tuple(find_org_nodes[0])
            else:
                find_unique_org_nodes = merge_path(org_graph_dict, find_org_nodes, org_model_paths)
                org_node_tp = tuple(find_unique_org_nodes)
            if len(org_node_tp) == 1:
                node_maps[org_node_tp[0]] = proposal_del_modified_model_node_names
            else:
                node_maps[org_node_tp] = proposal_del_modified_model_node_names
            for i in org_node_tp:
                if i not in org_model_node_names:
                    continue
                org_model_node_names.remove(i)
            for i in proposal_del_modified_model_node_names:
                if i not in modified_model_node_names:
                    continue
                modified_model_node_names.remove(i)

    # 这里仍然保持严格检查：如果还有 org 节点没匹配上，说明确实对应不上，应该报错。
    if len(org_model_node_names) != 0 or len(modified_model_node_names) != 0:
        print("====== DEBUG REMAINING ORG NODES ======")
        print(org_model_node_names[:50])
        print("====== DEBUG REMAINING MODIFIED NODES ======")
        print(modified_model_node_names[:50])
        raise AssertionError("org_model_node_names and modified_model_node_names should be empty")

    return node_maps

def split_to_single_op(model_, sub_model_sv_dir=None):
    initi_names = [i.name for i in model_.graph.initializer]

    itensor_to_nodes = dict()
    for node in model_.graph.node:
        for i_ in node.input:
            if i_ in initi_names:
                continue
            if i_ not in itensor_to_nodes:
                itensor_to_nodes[i_] = []
            if node.name not in itensor_to_nodes[i_]:
                itensor_to_nodes[i_].append(node.name)

    otensor_to_nodes = dict()
    for node in model_.graph.node:
        for i_ in node.output:
            if i_ not in otensor_to_nodes:
                otensor_to_nodes[i_] = []
            if node.name not in otensor_to_nodes[i_]:
                otensor_to_nodes[i_].append(node.name)

    org_in_tensors = list(model_.graph.input)
    org_in_names = [i.name for i in org_in_tensors]
    org_out_tensors = list(model_.graph.output)
    org_out_names = [i.name for i in org_out_tensors]
    org_mid_tensors = list(model_.graph.value_info)
    org_mid_names = [i.name for i in org_mid_tensors]
    inits_tensors = list(model_.graph.initializer)
    inits_names = [i.name for i in inits_tensors]

    sub_models = []
    for idx, node in enumerate(model_.graph.node):
        single_model = onnx.ModelProto()
        single_model.CopyFrom(model_)
        single_model.graph.Clear()
        single_model.graph.name = model_.graph.name

        single_model.graph.node.append(node)

        for i in node.input:
            if i in org_in_names:
                single_model.graph.input.append(org_in_tensors[org_in_names.index(i)])
            elif i in inits_names:
                init_tensor = inits_tensors[inits_names.index(i)]
                if init_tensor not in single_model.graph.initializer:
                    single_model.graph.initializer.append(init_tensor)
            elif i in org_mid_names:
                single_model.graph.input.append(org_mid_tensors[org_mid_names.index(i)])
            elif i == '':
                continue
            else:
                raise Exception('{} {} error'.format(node.name, i))

        for i in node.output:
            if i in org_mid_names:
                single_model.graph.output.append(org_mid_tensors[org_mid_names.index(i)])
            elif i in org_out_names:
                single_model.graph.output.append(org_out_tensors[org_out_names.index(i)])
            else:
                raise Exception('{} {} error'.format(node.name, i))

        onnx.checker.check_model(single_model)

        if sub_model_sv_dir is not None:
            onnx.save_model(single_model, os.path.join(
                sub_model_sv_dir, 'sub_{}.onnx'.format(idx)))
        sub_models.append(single_model)

    return sub_models

def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def _parse_fp8_spec(fp8_meta):
    if fp8_meta is None:
        return {"exp_bits": 4, "mantissa_bits": 3}

    # support tuple / list like (base, exp_bits, mantissa_bits)
    if isinstance(fp8_meta, (list, tuple)):
        if len(fp8_meta) >= 3:
            return {"exp_bits": int(fp8_meta[1]), "mantissa_bits": int(fp8_meta[2])}
        if len(fp8_meta) == 2:
            return {"exp_bits": int(fp8_meta[0]), "mantissa_bits": int(fp8_meta[1])}

    if isinstance(fp8_meta, dict):
        return {
            "exp_bits": int(fp8_meta.get("exp_bits", 4)),
            "mantissa_bits": int(fp8_meta.get("mantissa_bits", 3))
        }

    # string meta like "E4M3" / "E5M2"
    if isinstance(fp8_meta, str):
        fmt = fp8_meta.lower()
        if "e5m2" in fmt:
            return {"exp_bits": 5, "mantissa_bits": 2}
        if "e4m3" in fmt:
            return {"exp_bits": 4, "mantissa_bits": 3}

    return {"exp_bits": 4, "mantissa_bits": 3}

def write_json(data_dict, sv_path):
    with open(sv_path, 'w') as f:
        json.dump(data_dict, f, indent=4)


def resolve_tensor_file(tensor_name, search_dirs, exts=(".bin", ".tensorproto", ".pb")):
    normalized_name = replace_special_charactor_inverse(tensor_name)
    candidates = []

    def _collect(path):
        if os.path.exists(path):
            candidates.append(path)

    for ext in exts:
        for name in [tensor_name, normalized_name, replace_special_charactor(tensor_name)]:
            for d in search_dirs:
                _collect(os.path.join(d, name + ext))

    # digits based mapping: PPQ_Variable_24 -> *00024*.tensorproto / *24.tensorproto
    digit_match = re.findall(r"(\d+)$", tensor_name)
    if digit_match:
        suffix = digit_match[-1]
        padded = suffix.zfill(5)
        patterns = [suffix, padded, f"_{suffix}", f"_{padded}", f"out_{suffix}", f"out_{padded}"]
        for d in search_dirs:
            files = glob(os.path.join(d, "*"))
            for f in files:
                base = os.path.basename(f)
                if any(p in base for p in patterns):
                    if os.path.splitext(base)[1] in exts:
                        candidates.append(f)

    if candidates:
        # prefer tensorproto over bin to leverage metadata, and newest file if multiple
        candidates = sorted(candidates, key=lambda p: (not p.endswith('.tensorproto'), -os.path.getmtime(p)))
        return candidates[0]

    return None


def tensorproto_to_ndarray(path, fp8_meta=None):
    tensor = onnx.TensorProto()
    with open(path, "rb") as f:
        tensor.ParseFromString(f.read())

    try:
        data = numpy_helper.to_array(tensor)
        if data.dtype == np.float16 or data.dtype == np.float32:
            return data
        return data.astype(np.float32)
    except Exception:
        pass

    raw = tensor.raw_data
    if raw and len(raw):
        # 优先按 TensorProto 的显式类型解析，避免把 FP16/FP32 误解码成 FP8
        dtype_map = {
            onnx.TensorProto.FLOAT: np.float32,
            onnx.TensorProto.FLOAT16: np.float16,
            onnx.TensorProto.BFLOAT16: np.float16,
            onnx.TensorProto.UINT8: np.uint8,
            onnx.TensorProto.INT8: np.int8,
            onnx.TensorProto.UINT16: np.uint16,
            onnx.TensorProto.INT16: np.int16,
            onnx.TensorProto.INT32: np.int32,
            onnx.TensorProto.INT64: np.int64,
        }
        if tensor.data_type in dtype_map:
            return np.frombuffer(raw, dtype=dtype_map[tensor.data_type])

        # 其余情况视为 FP8 raw bytes，尝试按照 meta 解码
        spec = _parse_fp8_spec(fp8_meta)
        try:
            decoded = fp8_to_float16_bytes(raw, **spec)
            return np.frombuffer(decoded, dtype=np.float16)
        except Exception:
            return np.frombuffer(raw, dtype=np.float32)

    # fallback on field data
    if len(tensor.float_data):
        return np.array(tensor.float_data, dtype=np.float32)
    if len(tensor.int32_data):
        return np.array(tensor.int32_data, dtype=np.int32)

    return None


def load_tensor_data(path, default_dtype=np.float16, fp8_meta=None):
    if path is None:
        return None

    ext = os.path.splitext(path)[1].lower()
    if ext in [".tensorproto", ".pb"]:
        return tensorproto_to_ndarray(path, fp8_meta)

    if ext == ".bin":
        if fp8_meta is not None:
            return read_fp8_file(path, fp8_meta)
        return np.fromfile(path, dtype=default_dtype)

    # unknown extension, try best-effort
    try:
        return np.fromfile(path, dtype=default_dtype)
    except Exception:
        return None

def infer_onnx_layerwise(model, res_sv_dir, input_dir, from_onnx_res=False):
    org_model = copy.deepcopy(model)

    layer_results = dict()

    output_names = sum([i.output[:] for i in model.graph.node], [])
    vs = {i.name: i for i in list(model.graph.value_info) + list(model.graph.output)}
    outputs = [vs[i] for i in output_names if i in vs]
    while (len(model.graph.output)):
        model.graph.output.pop()
    model.graph.output.extend(outputs)

    sess = ort.InferenceSession(model.SerializeToString())
    input_dict = dict()
    for idx, info in enumerate(sess.get_inputs()):
        norm_name = replace_special_charactor_inverse(info.name)
        if os.path.isdir(input_dir):
            if from_onnx_res:
                ty = np.float32
            else:
                ty = np.float16
            data = np.fromfile(os.path.join(input_dir, norm_name + '.bin'),
                               dtype=ty).reshape(info.shape).astype(np.float32)
        else:
            print('Error, no input file')
        sv_path = os.path.join(res_sv_dir, '{}.bin'.format(norm_name))
        data.tofile(sv_path)
        input_dict[info.name] = data
        layer_results[replace_special_charactor_inverse(info.name)] = sv_path
    outputs = sess.run(None, input_dict)

    for i, (out_info, array) in enumerate(zip(sess.get_outputs(), outputs)):
        norm_name = replace_special_charactor_inverse(out_info.name)
        sv_path = os.path.join(res_sv_dir, '{}.bin'.format(norm_name))
        array.astype(np.float32).tofile(sv_path)
        layer_results[norm_name] = sv_path

    return org_model, layer_results

def infer_onnx_model(model, onnx_result_sv_dir, onnx_layerwise_resluts, external_input=None):
    sess = ort.InferenceSession(model.SerializeToString())

    input_data = dict()

    for tnsr in sess.get_inputs():
        if external_input is None:
            data = np.random.random(tnsr.shape).astype(np.float32)
        else:
            # NOTE: 原始脚本这里有逻辑问题，但一般不会走到这个分支，保持最小修改
            data = np.random.random(tnsr.shape).astype(np.float32)

        input_data[tnsr.name] = data
        onnx_layerwise_resluts.update({
            tnsr.name: os.path.join(onnx_result_sv_dir, '{}.bin'.format(tnsr.name))
        })
    outputs = sess.run(None, input_data)

    for idx, (out_info, array) in enumerate(zip(sess.get_outputs(), outputs)):
        sv_path = os.path.join(onnx_result_sv_dir, '{}.bin'.format(idx))
        array.astype(np.float32).tofile(sv_path)
        onnx_layerwise_resluts.update({
            out_info.name: sv_path
        })

def get_parser():
    parser = argparse.ArgumentParser(description='generate model io')
    parser.add_argument('tnt_output', type=str,
                        help='path of tnt output')
    parser.add_argument('--output', type=str, default='',
                        help='path to compare results, if not set, will be tnt_output/compare_result')
    parser.add_argument('--onnx', type=str, default='',
                        help='original onnx model without quant')
    parser.add_argument('--board_out', type=str, default='',
                        help='runing on board')
    parser.add_argument('--max_depth', type=int, default=32,
                        help='max recursive depth')
    parser.add_argument('--mode', type=str, default='layerwise',
                        choices=['layerwise', 'graphwise'],
                        help='compare mode')
    return parser.parse_args()

def os_cmd(cmd):
    if not cmd:
        return

    print("[CMD]: ", cmd)
    process = subprocess.Popen(cmd, shell=True, cwd=cur_path)
    return_code = process.wait()

    if return_code != 0:
        print("cmd failed: %s" % cmd)

def run_cmd(cmd, iq_model_out, tfl_info, tensor_name_paths):
    os_cmd(cmd)
    for idx, i in enumerate(tfl_info['input']):
        tensor_name_paths[i] = \
            [os.path.join(iq_model_out, 'model_in_aic_tag_{}.bin'.format(idx)),
             tfl_info['input_shape'][idx]['dtype']]
    for idx, i in enumerate(tfl_info['output']):
        tensor_name_paths[i] = \
            [os.path.join(iq_model_out, 'model_out_aic_tag_{}.bin'.format(
                idx + len(tfl_info['input']) + 1)),
             tfl_info['output_shape'][idx]['dtype']]


def run_tnnsim_ais(subop_info, iq_sv_layer_results, org_onnx_layer_results):
    iq_json_dir = os.path.dirname(subop_info['sv'])

    for file_format in ['tlf', 'iq', 'sv', 'sv_ref']:
        in_aic_names = []
        temp_d = os.path.join(iq_json_dir, file_format)
        if os.path.exists(temp_d):
            shutil.rmtree(temp_d)
        os.makedirs(temp_d, exist_ok=True)

        for idx, (aic_tensor_name, onnx_tensor) in enumerate(
                zip(subop_info['input_tensor'], subop_info['input'])):
            in_aic_name = 'model_in_{}.bin'.format(aic_tensor_name)
            in_aic_path = os.path.join(iq_json_dir, file_format, in_aic_name)
            in_aic_names.append(in_aic_path)
            in_type = subop_info['input_shape'][idx]['dtype']
            onnx_tensor_name = list(onnx_tensor.keys())[0]
            onnx_tensor_type = onnx_tensor[onnx_tensor_name]

            if onnx_tensor_type == 'fp32':
                if onnx_tensor_name in org_onnx_layer_results:
                    shutil.copy(org_onnx_layer_results[onnx_tensor_name], in_aic_path)
                    if in_type == 'f16':
                        data = np.fromfile(in_aic_path, dtype=np.float32)
                        data.astype(np.float16).tofile(in_aic_path)
                    else:
                        AssertionError("not support type {}".format(in_type))
                else:
                    raise AssertionError("{} should be in org_onnx_layer_results".
                                         format(onnx_tensor_name))
            elif onnx_tensor_type == 'fp16':
                if onnx_tensor_name in iq_sv_layer_results:
                    file = iq_sv_layer_results[onnx_tensor_name][file_format]
                    if os.path.exists(file):
                        shutil.copy(file, in_aic_path)
                    else:
                        print("ERROR, file not exists")
                    if in_type != 'f16':
                        AssertionError("not support type {}".format(in_type))
                else:
                    raise AssertionError("{} should be in iq_sv_layer_results".
                                         format(onnx_tensor_name))
            else:
                raise AssertionError("only support fp32 and fp16")

        out_aic_names = []
        for aic_tensor_name, onnx_tensor_name in zip(
                subop_info['output_tensor'], subop_info['output']):
            out_aic_name = 'model_out_{}'.format(aic_tensor_name)
            out_aic_names.append(os.path.join(iq_json_dir, file_format, out_aic_name))

        if file_format in ['iq', 'sv', 'sv_ref']:
            if len(in_aic_names) > 0:
                in_tensor_strs = os.path.join(iq_json_dir, file_format) + '/model_in_{NAME}.bin'
            elif len(in_aic_names) == 1:
                in_tensor_strs = in_aic_names[0]
            else:
                raise AssertionError('input num should greather than 1')
        else:
            in_tensor_strs = ' '.join(in_aic_names)

        if file_format in ['iq', 'sv', 'sv_ref']:
            if len(out_aic_names) > 0:
                out_tensor_strs = os.path.join(iq_json_dir, file_format) + '/model_out_{NAME}'
            elif len(out_aic_names) == 1:
                out_tensor_strs = out_aic_names[0]
            else:
                raise AssertionError('output num should greather than 1')
        else:
            out_tensor_strs = ''

        cur_pwd = os.getcwd()

        cmd = "cd {}".format(iq_json_dir)
        os_cmd(cmd)

        if file_format in ['iq', 'sv', 'sv_ref']:
            tnn_sim_cmd = "tnn_sim {} {} {}".format(
                subop_info[file_format], in_tensor_strs, out_tensor_strs)
            if file_format != 'iq':
                tnn_sim_cmd += " -sv -hw"
            os_cmd(tnn_sim_cmd)
        else:
            ts_ais_cmd = "ts_ais {} -i {}".format(subop_info['tlf'], in_tensor_strs)
            os_cmd(ts_ais_cmd)
        print('*' * 20)

        os_cmd("cd {}".format(cur_pwd))

        if file_format in ['iq', 'sv', 'sv_ref']:
            out_prefix = 'model'
        else:
            out_prefix = 'ais'
        for aic_tensor_name, onnx_tensor_name in zip(
                subop_info['output_tensor'], subop_info['output']):
            if onnx_tensor_name not in iq_sv_layer_results:
                iq_sv_layer_results[onnx_tensor_name] = dict()
            iq_sv_layer_results[onnx_tensor_name][file_format] = \
                os.path.join(iq_json_dir, file_format, '{}_out_{}.bin'.
                             format(out_prefix, aic_tensor_name))

def calc_simarity(ref, board, name='board'):
    simarity = 0.
    ref_range_str = ','
    board_range_str = ','
    if ref is None or board is None:
        return simarity, ref_range_str, board_range_str

    ref = np.asarray(ref).astype(np.float32)
    board = np.asarray(board).astype(np.float32)
    if ref.size == 0 or board.size == 0:
        return simarity, ref_range_str, board_range_str

    min_len = min(ref.size, board.size)
    if min_len != ref.size or min_len != board.size:
        ref = ref.flatten()[:min_len]
        board = board.flatten()[:min_len]

    try:
        simarity = 1. - cosine(
            ref.flatten().astype(np.float32), board.flatten().astype(np.float32))
        ref_range_str = "ref min:{} max:{}".format(ref.min(), ref.max())
        board_range_str = "{} min:{} max:{}".format(name, board.min(), board.max())
        if np.isnan(board).any() or np.isnan(ref).any():
            simarity = 'nan'
    except Exception as e:
        print(type(e).__name__, e)

    return simarity, ref_range_str, board_range_str

def is_bitmatch(data1, data2):
    assert data1.dtype == data2.dtype, 'should be same type'

    return (data2 == data1).all()

def get_pair_files(tlf_path):
    out_dir = os.path.dirname(os.path.dirname(tlf_path))

    info = {
        'tlf': tlf_path,
        'iq': os.path.join(
            os.path.dirname(out_dir), out_dir, os.path.basename(out_dir) + '.json'),
        'sv': os.path.join(
            os.path.dirname(out_dir), out_dir, os.path.basename(out_dir) + '_sv.json'),
        'sv_ref': os.path.join(
            os.path.dirname(out_dir),
            out_dir,
            'out',
            os.path.basename(out_dir) + '_sv_ref_model.json'),
    }

    return info


def export_xml(data_full, output_file):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import os

    data = {k: v for k, v in data_full.items() if len(v)}

    # process data
    nodes = list(data.keys())
    comparisons = [
        "onnx-iq",
        "iq-sv",
        "sv-sv_ref",
        "sv-ais",
        "sv-board"
    ]
    num_comparisons = len(comparisons)

    # color map
    color_map = {
        "onnx-iq": "#36D399",
        "iq-sv": "#967ADC",
        "sv-sv_ref": "#165DFF",
        "sv-ais": "#FF4B4B",
        "sv-board": "#FFC269"
    }

    # make subplots
    fig = make_subplots(
        rows=num_comparisons,
        cols=1,
        subplot_titles=comparisons,
        vertical_spacing=0.12,
        shared_xaxes=False
    )

    for i, comparison in enumerate(comparisons, 1):
        if comparison == "sv-sv_ref":
            # BITMATCH/NOT BITMATCH
            values = [1 if data[node][comparison] == "BITMATCH" else 0 for node in nodes]
            y_title = f"{comparison} Result"
            y_range = [0, 1.1]
            y_tickvals = [0, 1]
            y_ticktext = ["NOT BITMATCH", "BITMATCH"]
            hover_template = f"Tensor: %{{customdata}}<br>{comparison}: %{{y}}<extra></extra>"
        else:
            values = [data[node][comparison] for node in nodes]
            y_title = f"{comparison} Similarity"
            y_range = [0, 1.1]
            y_tickvals = None
            y_ticktext = None
            hover_template = f"Tensor: %{{customdata}}<br>{comparison}: %{{y:.3f}}<extra></extra>"

        fig.add_trace(
            go.Bar(
                x=list(range(len(nodes))),
                y=values,
                name=comparison,
                marker_color=color_map[comparison],
                hovertemplate=hover_template,
                customdata=nodes,
                width=0.6,
                showlegend=(i == 1)
            ),
            row=i,
            col=1
        )

        fig.update_xaxes(
            title_text="Output Tensor Names",
            tickvals=list(range(len(nodes))),
            ticktext=nodes,
            tickangle=-45,
            tickfont={"size": 10},
            row=i,
            col=1
        )

        fig.update_yaxes(
            title_text=y_title,
            range=y_range,
            tickvals=y_tickvals,
            ticktext=y_ticktext,
            title_font={"color": color_map[comparison]},
            row=i,
            col=1
        )

    min_chart_len = 30
    max_chart_len = max([len(i) for i in data])
    scale = max(1.0, max_chart_len / min_chart_len / 1.3)
    # set layout
    fig.update_layout(
        title={
            "text": "Layerwise Comparison Results",
            "font": {"size": 24},
            "x": 0.5,
            "xanchor": "center"
        },
        height=int(400 * scale * num_comparisons),
        width=1800,
        margin=dict(b=220, t=150),
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5
        }
    )

    fig.write_html(output_file, include_plotlyjs="cdn")
    print(f"Comparison charts saved to: {os.path.abspath(output_file)}")


def fp8_to_float16_bytes(fp8_bytes, base=3, exp_bits=4, mantissa_bits=3):
    float16_bytes = bytearray()
    exp_bias = (1 << (exp_bits - 1)) - 1

    for b in fp8_bytes:
        # parase E4M3 fp8
        sign = (b >> 7) & 0x01  # signed
        exponent = (b >> mantissa_bits) & ((1 << exp_bits) - 1)  # exp
        mantissa = b & ((1 << mantissa_bits) - 1)  # mantissa_bits

        # convert to float16, handle zero/denorm as 0
        float16_exp_bias = 15  # float16 stand offset is 15, fp8 is 7
        if exponent == 0:
            new_exponent = 0
            new_mantissa = 0 if mantissa == 0 else mantissa << (10 - mantissa_bits)
        else:
            new_exponent = exponent - exp_bias + float16_exp_bias
            # expand mantissa 3 -> 10
            new_mantissa = mantissa << (10 - mantissa_bits)

        float16_val = (sign << 15) | (new_exponent << 10) | new_mantissa

        float16_bytes.extend(struct.pack('<H', float16_val))

    return bytes(float16_bytes)


def read_fp8_file(path, fp8_meta=None):
    with open(path, 'rb') as f:
        fp8_bytes = f.read()

    spec = _parse_fp8_spec(fp8_meta)
    float16_bytes = fp8_to_float16_bytes(
        fp8_bytes,
        exp_bits=spec.get("exp_bits", 4),
        mantissa_bits=spec.get("mantissa_bits", 3)
    )
    data = np.frombuffer(float16_bytes, dtype=np.float16)

    return data


def run():
    global MAX_DEPTH
    args = get_parser()
    compare_mode = args.mode
    tnt_output = args.tnt_output
    MAX_DEPTH = args.max_depth
    contexts = os.listdir(tnt_output)
    modified_onnx_model = ''

    # 主 onnx 使用 output 目录名推断，避免乱选
    model_name = os.path.basename(tnt_output.rstrip('/'))
    cand = os.path.join(tnt_output, model_name + '.onnx')
    if os.path.exists(cand):
        modified_onnx_model = cand
    else:
        # 回退策略：目录中唯一的 .onnx
        for c in contexts:
            if os.path.isfile(os.path.join(tnt_output, c)) and c.endswith('.onnx'):
                modified_onnx_model = os.path.join(tnt_output, c)
        print("WARNING: fallback onnx:", modified_onnx_model)

    org_onnx_model = args.onnx
    assert os.path.exists(org_onnx_model), "The original model is essential."
    assert os.path.exists(modified_onnx_model), "The modified model is essential."

    assert shutil.which("tnn_sim"), 'tnn_sim not exists'
    assert shutil.which("ts_ais"), 'ts_ais not exists'

    print("Makesure tnn_sim ts_ais exists")
    print("run ts_dlc.py or imcc like this:")
    print("DLC_DEBUG=1 python tools/ts_dlc.py --onnx xxx.onnx --node-mem-max-size 0 ...")
    print("then run: python tools/gen_refgolden.py --onnx xxx.onnx --model_out path/tp/output")

    model_name = os.path.basename(modified_onnx_model)[:-len('.onnx')]

    idx_file = os.path.join(tnt_output, 'subnet_info.csv')
    model_bin_file = os.path.join(tnt_output, model_name + '.onnx')
    partition_info_file = os.path.join(tnt_output, 'partition_info.json')

    assert os.path.exists(idx_file), 'subnet_info.csv not exists'
    assert os.path.exists(model_bin_file), model_name + '.onnx not exists'
    assert os.path.exists(partition_info_file), 'partition_info_file.json not exists'

    output_dir = os.path.join(tnt_output, 'compare_result')
    if args.output != "":
        os.makedirs(args.output, exist_ok=True)
    if os.path.exists(args.output):
        output_dir = os.path.join(args.output, 'compare_result')

    os.makedirs(output_dir, exist_ok=True)

    onnx_result_sv_dir = os.path.join(output_dir, 'onnx_res')
    os.makedirs(onnx_result_sv_dir, exist_ok=True)

    input_dir = os.path.join(args.tnt_output, 'input')
    assert os.path.exists(input_dir), 'input not exists'

    org_onnx_layer_results = dict()

    model = onnx.load(org_onnx_model)

    # some onnx model has some questions
    remove_initializer_from_input(model)
    remove_duplicate_initializer(model)

    org_model = copy.deepcopy(model)

    fp8_quant_node_outs = dict()
    if not is_cunstom_fp8_model(model):
        try:
            model, _ = onnxsim.simplify(model)
            model = onnx.shape_inference.infer_shapes(model)
        except Exception as e:
            print(type(e).__name__, e)

        sess = ort.InferenceSession(org_model.SerializeToString())
        for idx, info in enumerate(sess.get_inputs()):
            norm_name = replace_special_charactor_inverse(info.name)
            data = np.fromfile(os.path.join(input_dir, norm_name + '.bin'), dtype=np.float16)
            data.astype(np.float32).tofile(os.path.join(onnx_result_sv_dir, norm_name + '.bin'))

        if compare_mode == 'graphwise':
            org_model, org_onnx_layer_results = \
                infer_onnx_layerwise(org_model, onnx_result_sv_dir, input_dir)
        elif compare_mode == 'layerwise':
            sub_models = split_to_single_op(org_model, sub_model_sv_dir=None)
            for sub_model in sub_models:
                _, sub_res = infer_onnx_layerwise(
                    sub_model, onnx_result_sv_dir, onnx_result_sv_dir, from_onnx_res=True)
                org_onnx_layer_results.update(sub_res)
    else:
        input_info = OrderedDict()
        for i_ in model.graph.input:
            name, ty, shape = get_shape(i_)
            norm_name = replace_special_charactor_inverse(name)
            data = np.fromfile(os.path.join(input_dir, norm_name + '.bin'), dtype=np.float16)
            new_shape = copy.deepcopy(shape)
            if len(shape) == 4:
                # process input data with stride
                if shape[-1] < 16:
                    new_shape[-1] = 16
                elif shape[-1] % 16:
                    new_shape[-1] = (shape[-1] % 16 + 1) * 16
                data = np.reshape(data, new_shape)[:, :, :, :shape[-1]]
            input_info[name] = data.astype(np.float32)

        layer_ress, fp8_model, _ = infer_onnx_with_custom_quant(
            org_model, external_data=input_info)
        fp8_quant_node_outs = get_model_quant_attrs(fp8_model)
        for layere_name, layer_res in layer_ress.items():
            norm_layere_name = replace_special_charactor_inverse(layere_name)
            sv_path = os.path.join(onnx_result_sv_dir, '{}.bin'.format(norm_layere_name))
            layer_res.astype(np.float32).tofile(sv_path)
            org_onnx_layer_results.update({norm_layere_name:
                                           sv_path})

    write_json(org_onnx_layer_results, os.path.join(
        onnx_result_sv_dir,
        'org_onnx_layer_results.json'))

    modified_model = onnx.load(modified_onnx_model)

    org_mode_node_to_modified_node = OrderedDict()
    org_model_info = None
    if compare_mode == 'layerwise':
        org_model_info = parse_model(org_model)
        modified_model_info = parse_model(modified_model)
        try:
            org_mode_node_to_modified_node = org_model_to_tlf_model(
                org_model_info, modified_model_info)
        except AssertionError as e:
            print("[WARN] layerwise mapping failed, fallback to graphwise:", e)
            compare_mode = 'graphwise'
            org_mode_node_to_modified_node = OrderedDict()

    tensor_json_file = os.path.join(output_dir, 'iq_sv_layer_results.json')
    iq_sv_layer_results = dict()
    cpu_node_otensors = [replace_special_charactor_inverse(i.name) for i in org_model.graph.input]

    node_in_outs = dict()
    modified_model_name_to_npu_id = dict()
    for node in modified_model.graph.node:
        if node.op_type != 'ArtFusedCallback':
            cpu_node_otensors.extend(list(node.output))
            continue
        attr_str = node.attribute[0].s.decode()
        if 'npu_id' in attr_str:
            npu_idx = attr_str.index("'npu_id': ")
            lf = attr_str[npu_idx:].index('[')
            rf = attr_str[npu_idx:].index(']')
            npu_id_str = attr_str[npu_idx:][lf + 1: rf]
        else:
            raise AssertionError("not support")
        node_in_outs[npu_id_str] = {
            'input': list(node.input),
            'output': list(node.output),
            'node_name': node.name
        }

        modified_model_name_to_npu_id[node.name] = npu_id_str

    tlf_exec_order_and_paths = OrderedDict()
    with open(idx_file, 'r') as f:
        for i in [i.strip() for i in f.readlines()]:
            if ',' not in i:
                continue
            idx_str, path = i.split(',')
            tlf_exec_order_and_paths[idx_str] = path

    for t in org_model.graph.input:
        t_name = replace_special_charactor_inverse(t.name)
        iq_sv_layer_results[t_name] = org_onnx_layer_results[t_name]

    # layerwise mode need group tlf_exec_order_and_paths
    tlf_exec_orders = OrderedDict()
    if compare_mode == 'layerwise':
        for org_node_names, modified_node_names in org_mode_node_to_modified_node.items():
            sub_orders = []
            for node_name in modified_node_names:
                # ignore cpu nodes
                if node_name not in modified_model_name_to_npu_id:
                    continue
                npu_id = modified_model_name_to_npu_id[node_name]
                if npu_id in tlf_exec_order_and_paths:
                    sub_orders.append(tlf_exec_order_and_paths[npu_id])
            tlf_exec_orders[org_node_names] = sub_orders
    else:
        # graphwise: 不依赖 org_node_names，直接按 npu_id 顺序执行
        tlf_exec_orders = {idx: [path] for idx, path in tlf_exec_order_and_paths.items()}

    processed_tlfs = []
    tlf_infos = []
    for org_node_names, tlf_paths in tlf_exec_orders.items():
        for idx, tlf_path in enumerate(tlf_paths):
            pairs = get_pair_files(tlf_path)
            data = read_json(pairs['iq'])
            tlf_info = pairs
            input_ty = []
            for i in data['input_tensor']:
                input_with_type = dict()
                if compare_mode == 'layerwise':
                    # 只有 layerwise 模式才使用 org_model_info / 节点输入信息
                    if isinstance(org_node_names, tuple):
                        node_in_tnsrs = []
                        for org_node_name in org_node_names:
                            node_in_tnsrs.extend(org_model_info['node_i_tnsrs'][org_node_name])
                    else:
                        node_in_tnsrs = org_model_info['node_i_tnsrs'][org_node_names]
                    use_onnxref_input = i in node_in_tnsrs
                else:
                    # graphwise 不依赖节点级拓扑，只按 CPU 输出 vs 中间张量区分类型
                    use_onnxref_input = False

                if i in cpu_node_otensors or use_onnxref_input:
                    input_with_type[i] = 'fp32'
                else:
                    input_with_type[i] = 'fp16'
                input_ty.append(input_with_type)
            tlf_info['input'] = input_ty
            tlf_info['output'] = data['output_tensor']
            tlf_info.update(data)
            if pairs['tlf'] not in processed_tlfs:
                processed_tlfs.append(pairs['tlf'])
                run_tnnsim_ais(tlf_info, iq_sv_layer_results, org_onnx_layer_results)
            tlf_infos.append(tlf_info)

    write_json(iq_sv_layer_results, tensor_json_file)

    compare_results = OrderedDict()
    for tnsr_name, infos in iq_sv_layer_results.items():
        if isinstance(infos, str):
            continue

        inverse_tnsr_name = replace_special_charactor(tnsr_name)
        if tnsr_name not in compare_results:
            compare_results[inverse_tnsr_name] = dict()

        if tnsr_name in org_onnx_layer_results:
            onnx_result = load_tensor_data(org_onnx_layer_results[tnsr_name], default_dtype=np.float32)
        else:
            continue

        fp8_meta = fp8_quant_node_outs.get(tnsr_name)

        tlf_out_path = infos.get('tlf')
        tlf_out_data = load_tensor_data(tlf_out_path, fp8_meta=fp8_meta)

        sv_out_path = infos['sv']
        sv_out_data = load_tensor_data(sv_out_path, fp8_meta=fp8_meta)

        iq_out_path = infos['iq']
        iq_out_data = None
        if os.path.exists(iq_out_path):
            iq_out_data = load_tensor_data(iq_out_path, fp8_meta=fp8_meta)

        sv_ref_out_path = infos['sv_ref']
        sv_ref_out_data = load_tensor_data(sv_ref_out_path, fp8_meta=fp8_meta)

        board_search_dirs = [d for d in [args.board_out, os.path.join(args.tnt_output, 'debug_out')] if d]
        board_out_path = resolve_tensor_file(tnsr_name, board_search_dirs)
        board_out_data = load_tensor_data(board_out_path, fp8_meta=fp8_meta)

        shape = []
        for tlf_info in tlf_infos:
            for idx, i in enumerate(tlf_info['output']):
                if i == tnsr_name:
                    shape = tlf_info['output_shape'][idx]['dim']
        if iq_out_data is not None:
            sim_res = calc_simarity(onnx_result, iq_out_data, name='onnx-iq')
            compare_results[inverse_tnsr_name]['onnx-iq'] = sim_res[0]
            if (isinstance(sim_res[0], float) and sim_res[0] < 0.99):
                print(inverse_tnsr_name)
            compare_results[inverse_tnsr_name]['onnx_range'] = sim_res[1]
            compare_results[inverse_tnsr_name]['iq_range'] = sim_res[2]
        else:
            compare_results[inverse_tnsr_name]['onnx-iq'] = 0.
        compare_results[inverse_tnsr_name]['iq-sv'] = calc_simarity(
            onnx_result, sv_out_data, name='iq-sv')[0]
        if len(shape):
            shape_ = copy.deepcopy(shape)
            shape_[-1] = -1
            if tlf_out_data is not None:
                tlf_out_data_ = np.asarray(tlf_out_data).reshape(shape_)[
                    :shape[0], :shape[1], :shape[2], :shape[3]]
                # keep dtype aligned before bitmatch
                if sv_ref_out_data is not None and sv_out_data is not None and sv_out_data.dtype != sv_ref_out_data.dtype:
                    sv_ref_out_data = sv_ref_out_data.astype(sv_out_data.dtype)
                compare_results[inverse_tnsr_name]['sv-sv_ref'] = \
                    "BITMATCH" if (sv_out_data is not None and sv_ref_out_data is not None and is_bitmatch(sv_out_data, sv_ref_out_data)) else 'NOT BITMATCH'
                sim_res = calc_simarity(sv_out_data, tlf_out_data_, name='sv-ais')
                compare_results[inverse_tnsr_name]['sv-ais'] = sim_res[0]
                compare_results[inverse_tnsr_name]['sv_range'] = sim_res[1]
                compare_results[inverse_tnsr_name]['ais_range'] = sim_res[2]
                if board_out_data is not None:
                    board_out_data_ = np.asarray(board_out_data).reshape(shape_)[
                        :shape[0], :shape[1], :shape[2], :shape[3]]
                    compare_results[inverse_tnsr_name]['sv-board'] = calc_simarity(
                        sv_out_data, board_out_data_, name='sv-board')[0]
                else:
                    compare_results[inverse_tnsr_name]['sv-board'] = 0.
            else:
                compare_results[inverse_tnsr_name]['sv-sv_ref'] = 'NOT BITMATCH'
                compare_results[inverse_tnsr_name]['sv-ais'] = 0.
                compare_results[inverse_tnsr_name]['sv-board'] = 0.

    write_json(compare_results, os.path.join(
        output_dir, '{}-compare_result_{}.json'.format(model_name, compare_mode)))

    export_xml(compare_results, os.path.join(
        output_dir, '{}-compare_result_{}.html'.format(model_name, compare_mode)))

    print('done.')

    failed_cases = []
    for tnsr_name, info in compare_results.items():
        if len(info) == 0:
            continue
        if info.get('sv-sv_ref', 'BITMATCH') == 'NOT BITMATCH' or info.get('sv-ais', 1.0) < 0.99:
            failed_cases.append([tnsr_name, info])

    return failed_cases

if __name__ == '__main__':
    run()
