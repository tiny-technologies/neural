project('neural', 'c', version: '0.1')

cc = meson.get_compiler('c')
math_dep = cc.find_library('m')

omp_dep = dependency('openmp')

executable(
    'neural',
    'src/main.c',
    dependencies: [math_dep],
    install : true,
)

executable(
    'neural-test',
    'src/test.c',
    dependencies: [math_dep],
    install : true,
)
