from codecarbon import OfflineEmissionsTracker, EmissionsTracker
import json
import pandas as pd
import re
import os
import numpy as np
import time
import psutil
import cpuinfo
import ast
from IPython.display import display



class Tracker:  

    def __init__(self, experimentid, filename, dataset_path, code_blocks_to_exclude):
        self.tracker = EmissionsTracker()
        self.experimentid = experimentid
        self.final_data = {}
        self.dataset_path = dataset_path
        self.code_blocks_to_exclude = code_blocks_to_exclude

        self.notebook_path = str(filename)

        with open(self.notebook_path, 'r', encoding='utf-8') as f:
            self.notebook = json.load(f)

        
    def __estimate_cpu_flops(self):
        # Estimate theoretical peak FLOPS for the CPU.
        cpu_info = psutil.cpu_freq() #cpuinfo.get_cpu_info()
        
        #print(cpu_info.keys())
            
        # Get number of physical cores
        num_cores = psutil.cpu_count(logical=False)
    
        # Get clock speed in Hz
        if cpu_info is not None:
            clock_speed = cpu_info.current #cpu_info['hz_actual'][0]  # In Hz
    
        # Estimate FLOP per cycle (simplified: assuming 8 for modern CPUs)
        flops_per_cycle = 8  # Conservative estimate (varies by architecture)
    
        # Compute estimated FLOPS
        estimated_flops = num_cores * clock_speed * flops_per_cycle
        
        # Convert to GFLOPS (billion FLOPS)
        estimated_gflops = estimated_flops / 1e9
        
        return f"{estimated_gflops:.2f}"
    
    def __measure_actual_flops(self):
        # Measure actual FLOPS using matrix multiplication.
        N = 1000  # Matrix size (increase for better measurement)
        A = np.random.rand(N, N)
        B = np.random.rand(N, N)
        
        start_time = time.time()
        C = np.dot(A, B)  # Perform matrix multiplication
        end_time = time.time()
        
        # Number of floating point operations (approx: 2*N^3)
        num_operations = 2 * (N ** 3)
        
        # Time taken
        elapsed_time = end_time - start_time
        
        # Compute FLOPS
        flops = num_operations / elapsed_time
        gflops = flops / 1e9  # Convert to GFLOPS
        
        return f"{gflops:.2f}"
    
    def __ram_measurement(self):
        ram_info = psutil.virtual_memory()

        total_ram = round(ram_info.total / (1024 ** 3), 2)
        used_ram = round(ram_info.used / (1024 ** 3), 2)
        available_ram = round(ram_info.available / (1024 ** 3), 2)
        ram_usage = ram_info.percent

        return {"Total_ram":total_ram,
                "Used_ram": used_ram,
                "Avaialble_ram": available_ram,
                "Ram_Usage": ram_usage}
    
    def __measure_disk_speed(self):
        test_file = "disk_speed_test.tmp"
        data = os.urandom(100 * 1024 * 1024)  # 100MB of random data

        # Measure write speed
        start_time = time.time()
        with open(test_file, "wb") as f:
            f.write(data)
        write_time = time.time() - start_time
        write_speed = (100 / write_time) if write_time > 0 else 0  # MB/s

        # Measure read speed
        start_time = time.time()
        with open(test_file, "rb") as f:
            f.read()
        read_time = time.time() - start_time
        read_speed = (100 / read_time) if read_time > 0 else 0  # MB/s

        # Clean up
        os.remove(test_file)

        return {"Write_Speed": write_speed,
                "Read_Speed": read_speed}

    def __count_dataframes_in_notebook(self, notebook):
    
        dataframe_patterns = [
            r"(\w+)\s*=\s*pd\.DataFrame\(",
            r"(\w+)\s*=\s*pd\.read_csv\(",
            r"(\w+)\s*=\s*pd\.read_excel\(",
            r"(\w+)\s*=\s*pd\.read_json\(",
            r"(\w+)\s*=\s*pd\.read_parquet\(",
            r"(\w+)\s*=\s*pd\.read_sql\(",
            r"(\w+)\s*=\s*pl\.read_csv\(",
            r"(\w+)\s*=\s*pl\.read_json\("
        ]
        
        dataframe_count = 0
        
        for cell in notebook.get("cells", [])[self.code_blocks_to_exclude:-self.code_blocks_to_exclude]:
            if cell.get("cell_type") == "code":
                code = "".join(cell.get("source", []))
                
                for pattern in dataframe_patterns:
                    matches = re.findall(pattern, code)
                    dataframe_count += len(matches)

        return dataframe_count
    
    def __detect_file_extensions_in_notebook(self, notebook):
        file_extensions = set()
        
        for cell in notebook.get("cells", [])[self.code_blocks_to_exclude:-self.code_blocks_to_exclude]:
            if cell.get("cell_type") == "code":
                code = "".join(cell.get("source", []))
                
                file_paths = re.findall(r'["\']([^"\']+\.(csv|json|xlsx|parquet|tsv|txt|xml|sql|gz))["\']', code)
                
                for file_path_tuple in file_paths:
                    file_name = os.path.basename(file_path_tuple[0])
                    file_extensions.add(file_name)
        
        return file_extensions
    
    def __assess_dataframe_structure(self):
        total_rows = 0
        total_columns = 0
        dataset_info = {}

        count = 0

        for file_path in self.dataset_path:
            try:
                df = pd.read_csv(file_path, sep=None, engine="python")

                dataset_name = os.path.basename(file_path)

                num_rows, num_columns = df.shape
                total_rows += num_rows
                total_columns += num_columns

                dataset_info[dataset_name] = {
                    "rows": num_rows,
                    "columns": num_columns,
                }

                count+=1

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        return {
            "total_rows": total_rows,
            "total_columns": total_columns,
            "datafiles": count
        } if dataset_info else "No datafiles detected or structures are empty."
    
    def __detectlib(self, notebook):
    
        set_of_lib = set()

        for cell in notebook.get('cells', [])[self.code_blocks_to_exclude:-self.code_blocks_to_exclude]:
            if cell.get('cell_type') == 'code':
                for line in cell.get('source', []):
                    matches = re.findall(r'^\s*(?:import|from) ([\w\.]+)', line)
                if matches:
                    for match in matches:
                        lib_name = match.split('.')[0]
                        set_of_lib.add(lib_name) 
        
        return {
            "num_libs": len(set_of_lib),
            "libs": set_of_lib}
    
    def __count_imports_in_notebook(self, notebook):
        import_count = 0
        notebook_content = None

        try:
            if isinstance(notebook, str) and os.path.exists(notebook):
                with open(notebook, 'r', encoding='utf-8') as notebook_file:
                    notebook_content = json.load(notebook_file)
            elif isinstance(notebook, dict):
                notebook_content = notebook
            else:
                raise ValueError("Invalid input.")

            for cell in notebook_content.get('cells', [])[self.code_blocks_to_exclude:-self.code_blocks_to_exclude]:
                if cell['cell_type'] == 'code':
                    for line in cell['source']:
                        if re.match(r'^\s*(import|from)\s+', line):
                            import_count += 1

        except Exception as e:
            print(f"Error reading the notebook: {e}")
        
        return import_count
    
    def __get_kernel_consumption(self):
        try:
            # Get the current process ID (PID)
            current_pid = os.getpid()

            # Find the kernel process
            kernel_process = None
            for process in psutil.process_iter(attrs=['pid', 'name']):
                try:
                    if process.info['pid'] == current_pid or 'python' in process.info['name'].lower():
                        kernel_process = process
                        break  # We assume the first matching process is the kernel
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

            if kernel_process is None:
                return "Kernel process not found."

            # Get CPU and memory usage
            cpu_usage = kernel_process.cpu_percent(interval=1)  # CPU usage after 1-second measurement
            memory_usage = kernel_process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB

            return {"CPU Usage (%)": cpu_usage, "Memory Usage (MB)": round(memory_usage, 2)}

        except Exception as e:
            return f"Error retrieving kernel consumption: {e}"
        
    def __count_algorithms_in_notebook(self, notebook):
        try:
            with open(notebook, 'r', encoding='utf-8') as notebook_file:
                notebook_content = json.load(notebook_file)
            
            # Patterns to detect algorithms
            function_def_pattern = r"^\s*def\s+\w+\s*\("
            loop_pattern = r"^\s*(for|while)\s+"
            function_call_pattern = r"^\s*\w+\s*\(.*\)"  # Simplified function call pattern
            
            function_count = 0
            loop_count = 0
            function_call_count = 0
            
            for cell in notebook_content.get("cells", [])[self.code_blocks_to_exclude:-self.code_blocks_to_exclude]:
                if cell.get("cell_type") == "code":
                    for line in cell.get("source", []):
                        if re.match(function_def_pattern, line):
                            function_count += 1
                        elif re.match(loop_pattern, line):
                            loop_count += 1
                        elif re.match(function_call_pattern, line) and not re.match(function_def_pattern, line):
                            function_call_count += 1
            
            total_algorithms = function_count + loop_count + function_call_count
            return {
                "Function Definitions": function_count,
                "Loops": loop_count,
                "Function Calls": function_call_count,
                "Total Algorithms": total_algorithms
            }
        
        except Exception as e:
            return {
                "Function Definitions": 0,
                "Loops": 0,
                "Function Calls": 0,
                "Total Algorithms": 0,
                "Error": str(e)
            }
        
    def __count_classification_algorithms(self, notebook):
        # Scans a Jupyter notebook for classification algorithms and counts them.
        
        # It detects classifiers from libraries such as:
        # - Scikit-learn
        # - XGBoost
        # - LightGBM
        # - TensorFlow/Keras
        # - PyTorch
        
        # List of common classification algorithms
        classification_keywords = [
            # Scikit-learn classifiers
            r"\bLogisticRegression\(", r"\bRandomForestClassifier\(", r"\bSVC\(",
            r"\bDecisionTreeClassifier\(", r"\bKNeighborsClassifier\(", r"\bGaussianNB\(",
            r"\bGradientBoostingClassifier\(", r"\bAdaBoostClassifier\(", r"\bMLPClassifier\(",
            r"\bHistGradientBoostingClassifier\(", r"\bExtraTreesClassifier\(",
            
            # XGBoost
            r"\bXGBClassifier\(",
            
            # LightGBM
            r"\bLGBMClassifier\(",
            
            # TensorFlow/Keras models
            r"\bSequential\(", r"\bDense\(", r"\bConv2D\(", r"\bLSTM\(",
            
            # PyTorch classifiers
            r"\btorch\.nn\.Linear\(", r"\btorch\.nn\.Conv2d\(", r"\btorch\.nn\.LSTM\("
        ]
        
        # Load notebook content
        try:
            with open(notebook, 'r', encoding='utf-8') as notebook_file:
                notebook_content = json.load(notebook_file)
            
            classification_count = 0

            # Scan each cell in the notebook
            for cell in notebook_content.get("cells", [])[self.code_blocks_to_exclude:-self.code_blocks_to_exclude]:
                if cell.get("cell_type") == "code":
                    code = "".join(cell.get("source", []))  # Get full cell code

                    # Check for classifier patterns
                    for pattern in classification_keywords:
                        matches = re.findall(pattern, code)
                        classification_count += len(matches)

            return classification_count

        except Exception as e:
            return {"Error": f"Error reading the notebook: {e}"}
        
    def __count_regression_algorithms(self, notebook):
        # Scans a Jupyter notebook for regression algorithms and counts them.
        
        # It detects regressors from:
        # - Scikit-learn
        # - XGBoost
        # - LightGBM
        # - TensorFlow/Keras
        # - PyTorch
        
        # List of common regression algorithms
        regression_keywords = [
            # Scikit-learn regressors
            r"\bLinearRegression\(", r"\bRidge\(", r"\bLasso\(",
            r"\bElasticNet\(", r"\bSVR\(", r"\bDecisionTreeRegressor\(",
            r"\bRandomForestRegressor\(", r"\bGradientBoostingRegressor\(",
            r"\bAdaBoostRegressor\(", r"\bKNeighborsRegressor\(",
            r"\bMLPRegressor\(", r"\bExtraTreesRegressor\(",
            r"\bHistGradientBoostingRegressor\(",

            # XGBoost
            r"\bXGBRegressor\(",

            # LightGBM
            r"\bLGBMRegressor\(",

            # TensorFlow/Keras models
            r"\bSequential\(", r"\bDense\(",

            # PyTorch models
            r"\btorch\.nn\.Linear\(", r"\btorch\.nn\.Conv2d\("
        ]
        
        # Load notebook content
        try:
            with open(notebook, 'r', encoding='utf-8') as notebook_file:
                notebook_content = json.load(notebook_file)
            
            regression_count = 0

            # Scan each cell in the notebook
            for cell in notebook_content.get("cells", [])[self.code_blocks_to_exclude:-self.code_blocks_to_exclude]:
                if cell.get("cell_type") == "code":
                    code = "".join(cell.get("source", []))  # Get full cell code

                    # Check for regression patterns
                    for pattern in regression_keywords:
                        matches = re.findall(pattern, code)
                        regression_count += len(matches)

            return regression_count

        except Exception as e:
            return {"Error": f"Error reading the notebook: {e}"}

    def __compute_henry_kafura_metrics(self):
        function_calls = {}
        global_calls = set()

        for cell in self.notebook.get("cells", [])[self.code_blocks_to_exclude:-self.code_blocks_to_exclude]:
            if cell.get("cell_type") == "code":
                code = "".join(cell.get("source", []))
                try:
                    tree = ast.parse(code)
                except SyntaxError:
                    continue

                for node in ast.iter_child_nodes(tree):
                    # Track function definitions and their internal calls
                    if isinstance(node, ast.FunctionDef):
                        func_name = node.name
                        function_calls[func_name] = set()

                        for child in ast.walk(node):
                            if isinstance(child, ast.Call):
                                if isinstance(child.func, ast.Name):
                                    function_calls[func_name].add(child.func.id)
                    else:
                        # Also track top-level (global scope) function calls
                        for child in ast.walk(node):
                            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                                global_calls.add(child.func.id)

        # If nothing was found
        if not function_calls and not global_calls:
            return "Error: Could not retrieve function call data."

        # Add global scope calls as a synthetic "global" function
        if global_calls:
            function_calls["_global_scope_"] = global_calls

        # Compute Henry-Kafura metrics
        complexity_metrics = {}
        for function, calls in function_calls.items():
            F_in = sum(1 for funcs in function_calls.values() if function in funcs)
            F_out = len(calls)
            complexity_metrics[function] = {
                "F_in": F_in,
                "F_out": F_out,
                "Complexity": F_in * (F_out ** 2)
            }

        return complexity_metrics

    def __cyclomatic_complexity_ast(self, tree):
        # Computes Cyclomatic Complexity (CC) from an AST tree.
        nodes = 1  # Start with 1 (entry point)
        edges = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                nodes += 1
                edges += 2  # Each control structure introduces a new path
            elif isinstance(node, ast.IfExp):  # Ternary operator (e.g., x if cond else y)
                nodes += 1
                edges += 2
            elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.And, ast.Or)):
                edges += 1  # Boolean short-circuiting increases complexity

        complexity = edges - nodes + 2
        return max(complexity, 1)  # CC is at least 1

    def __compute_cyclomatic_complexity(self, notebook_path):
        # Computes total Cyclomatic Complexity across code cells in the notebook.
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        total_complexity = 0

        for cell in notebook.get("cells", [])[self.code_blocks_to_exclude:-self.code_blocks_to_exclude]:
            if cell["cell_type"] == "code":
                code = "".join(cell["source"])
                try:
                    tree = ast.parse(code)
                    total_complexity += self.__cyclomatic_complexity_ast(tree)
                except SyntaxError:
                    continue  # skip cells with invalid syntax

        return total_complexity
    
    def __count_executed_lines_of_code(self, notebook_path):
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        total_lines = 0

        for cell in notebook.get("cells", [])[self.code_blocks_to_exclude:-self.code_blocks_to_exclude]:
            if cell["cell_type"] == "code":
                code_lines = cell["source"]
                executable_lines = [
                    line for line in code_lines 
                    if line.strip() and not line.strip().startswith("#")  # Ignore empty lines and comments
                ]
                total_lines += len(executable_lines)

        return total_lines


    def __count_functions_in_notebook(self, notebook_path):
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        function_count = 0

        for cell in notebook.get("cells", [])[self.code_blocks_to_exclude:-self.code_blocks_to_exclude]:
            if cell["cell_type"] == "code":  # Only analyze code cells
                code = "".join(cell["source"])  # Get full code as a string
                try:
                    tree = ast.parse(code)  # Parse code into an AST
                    function_count += sum(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
                except SyntaxError:
                    pass  # Skip cells with invalid Python syntax

        return function_count

    def hello( self, name ):
        print( "Hello, " + name + "!. Welcome from Tracker package." )

    def version(self):
        from importlib.metadata import version
        return "lusofona-pckg v. " + version('lusofona-pckg')
    
    def start(self):
        self.tracker.start()

    def collect(self):

        ram_data = self.__ram_measurement()

        disk_speed = self.__measure_disk_speed()

        dataframe_structure = self.__assess_dataframe_structure()

        libraries = self.__detectlib(self.notebook)

        algorithms = self.__count_algorithms_in_notebook(self.notebook_path)

        code_carbon_data = json.loads(self.tracker.final_emissions_data.toJSON())

        hk_metrics = self.__compute_henry_kafura_metrics()

        cc_plus_data = {'cpu_flops': float(self.__estimate_cpu_flops()),
                    'actual_flops': float(self.__measure_actual_flops()),
                    'total_ram': ram_data["Total_ram"],
                    'used_ram': ram_data["Used_ram"],
                    'available_ram': ram_data["Avaialble_ram"],
                    'ram_usage': ram_data["Ram_Usage"],
                    'write_speed': disk_speed["Write_Speed"],
                    'read_speed': disk_speed["Read_Speed"],
                    'dataframe_count': self.__count_dataframes_in_notebook(self.notebook),
                    'file_extensions': self.__detect_file_extensions_in_notebook(self.notebook),
                    'total_rows': dataframe_structure["total_rows"],
                    'total_columns': dataframe_structure["total_columns"],
                    'datafiles': dataframe_structure["datafiles"],
                    'number_of_libraries': libraries['num_libs'],
                    'library_names': libraries['libs'],
                    'import_count': self.__count_imports_in_notebook(self.notebook),
                    'kernel_consumption': self.__get_kernel_consumption(),
                    'function_definitions': algorithms['Function Definitions'],
                    'loops': algorithms['Loops'],
                    'function_calls': algorithms['Function Calls'],
                    'total_algorithms': algorithms['Total Algorithms'],
                    'classification_algorithms': self.__count_classification_algorithms(self.notebook_path),
                    'regression_algorithms': self.__count_regression_algorithms(self.notebook_path),
                    'F_in': hk_metrics['_global_scope_']['F_in'],
                    'F_out': hk_metrics['_global_scope_']['F_out'],
                    'Complexity': hk_metrics['_global_scope_']['Complexity'],
                    'cyclomatic_complexity': self.__compute_cyclomatic_complexity(self.notebook_path),
                    'number_of_lines': self.__count_executed_lines_of_code(self.notebook_path),
                    'number_of_functions': self.__count_functions_in_notebook(self.notebook_path),
                    'region': code_carbon_data["region"],
                    'carbon_emissions_kg': code_carbon_data["emissions"],
                    'energy_consumed_kwh': code_carbon_data["energy_consumed"],
                    'duration_s': code_carbon_data["duration"],
                    'cpu_power_watt': code_carbon_data["cpu_power"],
                    'ram_power_watt': code_carbon_data["ram_power"],
                    'timestamp': code_carbon_data.get("timestamp"),
                    'project_name': "etracker", #code_carbon_data.get("project_name"),
                    'run_id': code_carbon_data.get("run_id"),
                    'experiment_id': self.experimentid, #code_carbon_data.get("experiment_id"),
                    'duration_s': code_carbon_data.get("duration"),
                    'carbon_emissions_kg': code_carbon_data.get("emissions"),
                    'emissions_rate': code_carbon_data.get("emissions_rate"),
                    'cpu_power_watt': code_carbon_data.get("cpu_power"),
                    'gpu_power_watt': code_carbon_data.get("gpu_power"),
                    'ram_power_watt': code_carbon_data.get("ram_power"),
                    'cpu_energy': code_carbon_data.get("cpu_energy"),
                    'gpu_energy': code_carbon_data.get("gpu_energy"),
                    'ram_energy': code_carbon_data.get("ram_energy"),
                    'energy_consumed_kwh': code_carbon_data.get("energy_consumed"),
                    'country_name': code_carbon_data.get("country_name"),
                    'country_iso_code': code_carbon_data.get("country_iso_code"),
                    'region': code_carbon_data.get("region"),
                    'cloud_provider': code_carbon_data.get("cloud_provider"),
                    'cloud_region': code_carbon_data.get("cloud_region"),
                    'os': code_carbon_data.get("os"),
                    'python_version': code_carbon_data.get("python_version"),
                    'codecarbon_version': code_carbon_data.get("codecarbon_version"),
                    'cpu_count': code_carbon_data.get("cpu_count"),
                    'cpu_model': code_carbon_data.get("cpu_model"),
                    'gpu_count': code_carbon_data.get("gpu_count"),
                    'gpu_model': code_carbon_data.get("gpu_model"),
                    'longitude': code_carbon_data.get("longitude"),
                    'latitude': code_carbon_data.get("latitude"),
                    'ram_total_size': code_carbon_data.get("ram_total_size"),
                    'tracking_mode': code_carbon_data.get("tracking_mode"),
                    'on_cloud': code_carbon_data.get("on_cloud"),
                    'pue': code_carbon_data.get("pue"),
                    }

        self.final_data = {**cc_plus_data}
        
        # write data to csv file if it does not exist , otherwise append to it
        
        df = pd.DataFrame([self.final_data])
        output_file = f"tracker_output.csv"
        if not os.path.exists(output_file):
            df.to_csv(output_file, index=False)
        else:
            df.to_csv(output_file, mode='a', header=False, index=False)

        
    def stop(self):
        self.tracker.stop()
        
        self.collect()

    def view(self):
        display(self.final_data)