# File with most basic functions

def hello( name ):
    print( "Hello, " + name + "!. Welcome to a Python Package Creation Example." )

def version():
    from importlib.metadata import version
    return "lusofona-pckg v. " + version('lusofona-pckg')
