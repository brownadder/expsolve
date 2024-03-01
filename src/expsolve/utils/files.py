import os

def localfilepath(filename, modulepath):
    # Get the path of the current module
    module_path = os.path.abspath(modulepath)

    # Construct the path to the file in the same folder
    file_path = os.path.join(os.path.dirname(module_path), filename)

    return file_path
