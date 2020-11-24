import os
import re


def main():
	"""
	Fixes the resources of all Flambe config files to point to the correct  
	absolute path on your machine.  Note that Flambe does not support 
	the use of local paths for resources, which is why this function is 
	necessary.

	"""
	base_dir = os.getcwd()
	iterator = os.walk('configs')
	for directory, _, files in iterator:
		for file in files:
			if '.yaml' in file:
				path = directory + '/' + file
				yaml_text = open(path, 'r').read()
				modified_text = replace_resources(yaml_text, base_dir)
				open(path, 'w').write(modified_text)


def replace_resources(yaml_text: str, base_dir: str) -> str:
	resource_paths = findall(yaml_text)
	for old_path in resource_paths:
		idx = old_path.find('/imitkd/') + len('/imitkd/')
		new_path = base_dir + '/' + old_path[idx:]
		yaml_text = yaml_text.replace(old_path, new_path)
	return yaml_text


def findall(text: str, pattern: str = '/imitkd/') -> str:
	pattern = r'(\S*%s\S*)' % pattern
	lst = re.findall(pattern, text)
	return lst


if __name__ == "__main__":
    main()
