import os
import sys
import re

import importlib.util
import inspect as module_inspect



def load_modules_config_dynamic(module_root):
    # module_root = "hehuprofiler/measurement"
    # module_name = "msei.py"

    modules_filename = os.listdir(module_root)

    modules_name_N_path = [(module_filename.replace(".py", ""), os.path.join(module_root, module_filename)) for module_filename in modules_filename]
    
    modules_config = {"modules":[]}

    for module_basename, module_path in modules_name_N_path:
        # print("module_basename: ", module_basename)
        if not re.match(r".+\.py$", module_path):
            continue

        spec = importlib.util.spec_from_file_location(module_basename, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class_names = module_inspect.getmembers(module, module_inspect.isclass)
        # print("class_names: ", class_names)
        class_name = None
        for c in class_names:
            c_name = c[0]
            if re.match(r"^(Segmentation|Measurement|MeasureImage|Explorer)\w+$", c_name):
                class_name = c_name
                break

        if class_name == None:
            continue
        # print("class_name: ", class_name)

        class_temp = getattr(module, class_name)

        # print("class_temp: ", class_temp)

        class_temp = class_temp()
        class_temp_disabled = class_temp.disabled
        class_temp_default = class_temp.default
        class_temp_name = class_temp.module_name
        class_temp_setting = class_temp.settings_
        try:
            class_temp_rank_type = class_temp.rank_type_activated
        except:
            class_temp_rank_type = []
        try:
            class_temp_object_type = class_temp.object_type_activated
        except:
            class_temp_object_type = []
        try:
            class_temp_object_parts = class_temp.object_parts_activated
        except:
            class_temp_object_parts = []
        try:
            class_temp_light_type = class_temp.light_type_activated
        except:
            class_temp_light_type = []



        del class_temp

        # del class_temp

        module_config = {
            "module_name": class_temp_name, 
            "module_setting": class_temp_setting, 
            "module_disabled": class_temp_disabled,
            "module_default": class_temp_default,
            "module_enabled": {
                "rank_type": class_temp_rank_type,
                "object_type": class_temp_object_type,
                "object_parts": class_temp_object_parts,
                "light_type": class_temp_light_type
            }
        }
        # print("module_config: ", module_config)
        modules_config["modules"].append(module_config)
        # print("modules_config: ", modules_config)

    return modules_config

def load_modules_dynamic(module_root, class_expected):
    # module_root = "hehuprofiler/measurement"
    # module_name = "msei.py"

    modules_filename = os.listdir(module_root)
    modules_name_N_path = [(module_filename.replace(".py", ""), os.path.join(module_root, module_filename)) for module_filename in modules_filename]
    
    for module_basename, module_path in modules_name_N_path:
        # print("module_basename: ", module_basename)
        if not re.match(r".+\.py$", module_path):
            continue

        spec = importlib.util.spec_from_file_location(module_basename, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class_names = module_inspect.getmembers(module, module_inspect.isclass)
        # print("class_names: ", class_names)
        class_name = None
        print("class_names: ", class_names)
        for c in class_names:
            c_name = c[0]
            if re.match(r"^(Segmentation|Measurement|MeasureImage|Explorer)\w+$", c_name):
                class_name = c_name
                break

        if class_name == None:
            continue

        class_temp = getattr(module, class_name)

        # print("class_temp: ", class_temp)

        class_temp_name = class_temp().module_name
        if class_temp_name == class_expected:

            print("Matched class: ", class_temp_name)

            return True, class_temp, "Success"

    return False, None, "No Match Class"