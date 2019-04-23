from distutils.core import setup, Extension

module = Extension('CAccel', sources = ["C_Accle.c"])

setup (name = 'CAccel',
       version = '1.0', 
       description = '',
       ext_modules = [module]
       )
