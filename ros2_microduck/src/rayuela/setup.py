import glob

from setuptools import find_packages, setup

package_name = 'rayuela'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='robot@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'sim_microduck = rayuela.run_env:main',
            'sim_node = rayuela.sim_node:main',
            'bridge_node = rayuela.bridge_node:main',
            'vision_node = rayuela.vision_node:main',
            'control_node = rayuela.control_node:main',
            'teleop_keyboard = rayuela.teleop_keyboard:main',
        ],
    },
)
