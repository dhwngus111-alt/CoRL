from setuptools import setup, find_packages

setup(
    name='risky_overcooked_py',
    version='1.1.0',
    description='Risky Overcooked environment (MDP, env, subgoal layouts) — '
                'water-slip risk + subgoal buttons. Adapted from overcooked_ai.',
    packages=find_packages(where='src',
                           include=['risky_overcooked_py', 'risky_overcooked_py.*']),
    package_dir={'': 'src'},
    package_data={'risky_overcooked_py': [
        'data/layouts/*.layout',
        'data/graphics/*', 'data/fonts/*', 'data/planners/*',
    ]},
    include_package_data=True,
    install_requires=['numpy', 'scipy', 'gym', 'gymnasium', 'pettingzoo',
                      'pygame', 'dill', 'tqdm', 'opencv-python', 'ipywidgets'],
    python_requires='>=3.8',
    url='https://github.com/idonggyu-L/overcook2',
    license='MIT',
)
