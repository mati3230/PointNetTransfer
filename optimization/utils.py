import numpy as np
import os
import tensorflow as  tf
import math
import json


def load_args_file(args_file, types_file):
    with open(args_file) as f:
        params = json.load(f)
    with open(types_file) as f:
        params_types = json.load(f)
    return params, params_types


def split_examples(train_idxs, train_n, client_id, n_clients):
    examples_per_client = math.floor(train_n / n_clients)
    start_i = examples_per_client * client_id
    stop_i = examples_per_client * (client_id + 1)
    if client_id == n_clients - 1:
        stop_i = train_n
    return train_idxs[start_i:stop_i]


def socket_send(file, sock, buffer_size=4096):
    f = open(file,"rb")
    l = f.read(buffer_size)
    while (l):
        sock.send(l)
        l = f.read(buffer_size)
    f.close()


def socket_recv(file, sock, buffer_size=4096, timeout=4, msg_size=None):
    f = open(file, "wb")
    if msg_size is None:
        sock.settimeout(None)
    recv_size = 0
    l = sock.recv(buffer_size)
    recv_size += len(l)
    if msg_size is None:
        sock.settimeout(timeout)
    while (True):
        f.write(l)
        if msg_size is None:
            try:
                l = sock.recv(buffer_size)
                recv_size += len(l)
            except:
                break
        else:
            # msg received?
            if recv_size == msg_size:
                break
            l = sock.recv(buffer_size)
            recv_size += len(l)
    f.close()
    if msg_size is None:
        sock.settimeout(None)
    return recv_size


def mkdir(directory):
    """Method to create a new directory.

    Parameters
    ----------
    directory : str
        Relative or absolute path.
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)


def file_exists(filepath):
    """Check if a file exists.

    Parameters
    ----------
    filepath : str
        Relative or absolute path to a file.

    Returns
    -------
    boolean
        True if the file exists.

    """
    return os.path.isfile(filepath)


def save_config(log_dir, config):
    """Save a custom configuration such as learning rate.

    Parameters
    ----------
    log_dir : str
        Directory where the configuration should be placed.
    config : str
        String with the configuration.
    """
    text_file = open(log_dir + "/config.txt", "w")
    text_file.write(config)
    text_file.close()


def importer(name, root_package=False, relative_globals=None, level=0):
    """Imports a python module.

    Parameters
    ----------
    name : str
        Name of the python module.
    root_package : boolean
        See https://docs.python.org/3/library/functions.html#__import__.
    relative_globals : type
        See https://docs.python.org/3/library/functions.html#__import__.
    level : int
        See https://docs.python.org/3/library/functions.html#__import__.

    Returns
    -------
    type
        Python module. See
        https://docs.python.org/3/library/functions.html#__import__.

    """
    return __import__(name, locals=None, # locals has no use
                      globals=relative_globals,
                      fromlist=[] if root_package else [None],
                      level=level)


def get_type(path_str, type_str):
    """Load a specific class type.

    Parameters
    ----------
    path_str : str
        Path to the python file of the desired class.
    type_str : str
        String of the class name.

    Returns
    -------
    type
        Requested class type.

    """
    module = importer(path_str)
    mtype = getattr(module, type_str)
    return mtype


def parse_float_value(usr_cmds, i, type):
    """Parse a float value from a user command list.

    Parameters
    ----------
    usr_cmds : list(str)
        List where float values are stored.
    i : int
        The i-th value should be the name of the float value. The (i+1)-th
        value should be the float value itself.
    type : str
        Should be whether:
            real pos float: > 0
            real neg float: < 0
            pos float: >= 0
            neg float: <= 0

    Returns
    -------
    tuple(float, str)
        Returns the float value and a string with 'ok' if no error occurs. In
        case of an error a tuple('error', error message) is returned.

    """
    try:
        value = usr_cmds[i + 1]
    except IndexError:
        return ("error", "No value")
    try:
        value = float(value)
    except ValueError:
        return ("error", "Value '" + str(value) + "' cannot be converted to " + str(type))
    if type == "real pos float" and value <= 0:
        return ("error", "Value has to be greater than 0")
    elif type == "real neg float" and value >= 0:
        return ("error", "Value has to be greater than 0")
    elif type == "pos float" and value < 0:
        return ("error", "Value has to be greater than or equal 0")
    elif type == "neg float" and value > 0:
        return ("error", "Value has to be greater than or equal 0")
    return (value, "ok")


def _parse_int_value(value, type):
    """Parse a int value.

    Parameters
    ----------
    value : int
        An int value.
    type : str
        real pos int: > 0
        real neg int: < 0
        pos int: >= 0
        neg int: <= 0

    Returns
    -------
    type
        Returns the int value and a string with 'ok' if no error occurs. In
        case of an error a tuple('error', error message) is returned.

    """
    try:
        value = int(value)
    except ValueError:
        return ("error", "Value '" + str(value) + "' cannot be converted to " + str(type))
    if type == "real pos int" and value <= 0:
        return ("error", "Value has to be greater than 0")
    elif type == "real neg int" and value >= 0:
        return ("error", "Value has to be greater than 0")
    elif type == "pos int" and value < 0:
        return ("error", "Value has to be greater than or equal 0")
    elif type == "neg int" and value > 0:
        return ("error", "Value has to be greater than or equal 0")
    return (value, "ok")


def parse_int_value(usr_cmds, i, type):
    """Parse a float value from a user command list.

    Parameters
    ----------
    usr_cmds : list(str)
        List where int values are stored.
    i : int
        The i-th value should be the name of the int value. The (i+1)-th
        value should be the int value itself.
    type : str
        real pos int: > 0
        real neg int: < 0
        pos int: >= 0
        neg int: <= 0

    Returns
    -------
    tuple(int, str)
        Returns the int value and a string with 'ok' if no error occurs. In
        case of an error a tuple('error', error message) is returned.

    """
    try:
        value = usr_cmds[i + 1]
    except IndexError:
        return ("error", "No value")
    return _parse_int_value(value, type)


def parse_bool_value(usr_cmds, i):
    """Parse a boolean value from a user command list.

    Parameters
    ----------
    usr_cmds : list(str)
        List where int values are stored.
    i : int
        The i-th value should be the name of the int value. The (i+1)-th
        value should be the int value itself.

    Returns
    -------
    tuple(boolean, str)
        Returns the boolean value and a string with 'ok' if no error occurs. In
        case of an error a tuple('error', error message) is returned.

    """
    try:
        value = usr_cmds[i + 1]
    except IndexError:
        return ("error", "No value - expected True or False")
    if value != "True" and value != "False":
        return ("error", "Invalid value - expected True or False")
    if value == "True":
        return (True, "ok")
    return (False, "ok")


def parse_list_int(usr_cmds, i):
    """Parse a list of int values from a user command list.

    Parameters
    ----------
    usr_cmds : list(str)
        List where int values are stored.
    i : int
        The i-th value should be the name of the int value. The (i+1)-th
        value should be the int value itself.

    Returns
    -------
    tuple(list(int), str)
        Returns the list(int) and a string with 'ok' if no error occurs. In
        case of an error a tuple('error', error message) is returned.

    """
    try:
        value = usr_cmds[i + 1]
    except IndexError:
        return "error", "No value - expected list"
    if value == "":
        return None, "ok"
    if len(value) == 1:
        return "error", "List should be at least '[]'"
    if len(value) == 2:
        return [], "ok"
    if len(value) >= 2:
        if value[0] != "[":
            return "error", "List should begin with ["
        if value[-1] != "]":
            return "error", "List should end with ]"
        l_values = value[1:-1]
        l_values = l_values.split(",")
        result = []
        for i in range(len(l_values)):
            l_val = l_values[i]
            if l_val == "" or l_val == " ":
                continue
            val = 0
            if l_val.isdigit():
                val = int(l_val)
            else:
                return "error", str(l_val) + " cannot be converted to int"
            result.append(val)
        return result, "ok"


def parse_list_tuple(usr_cmds, i):
    """Parse a list of tuple values from a user command list.

    Parameters
    ----------
    usr_cmds : list(str)
        List where int values are stored.
    i : int
        The i-th value should be the name of the int value. The (i+1)-th
        value should be the int value itself.

    Returns
    -------
    tuple(list(tuple), str)
        Returns the list(tuple) and a string with 'ok' if no error occurs. In
        case of an error a tuple('error', error message) is returned.

    """
    try:
        value = usr_cmds[i + 1]
    except IndexError:
        return "error", "No value - expected list of tuples"
    if value == "":
        return None, "ok"
    if len(value) == 1:
        return "error", "List should be at least '[]'"
    if len(value) == 2:
        return [], "ok"
    if len(value) >= 2:
        if value[0] != "[":
            return "error", "List should begin with ["
        if value[-1] != "]":
            return "error", "List should end with ]"
        l_values = value[1:-1]
        if l_values[0] != "(":
            return "error", "Tuple should begin with ("
        if l_values[-1] != ")":
            return "error", "Tuple should end with )"
        #print(l_values)
        start_idx = 0
        result = []
        while True:
            #print(start_idx)
            stop_idx = -1
            si = -1
            for i in range(start_idx, len(l_values)):
                if l_values[i] == "(":
                    si = i
                if l_values[i] == ")":
                    stop_idx = i
                if stop_idx != -1:
                    break
            if stop_idx <= si:
                return "error", "A tuple should at least have the form '(x, )'"
            tuple_str = l_values[si:stop_idx+1]
            t, msg = parse_tuple_int_value(["", tuple_str], 0, "pos int")
            if t == "error":
                return t, msg
            result.append(t)
            start_idx = stop_idx + 1
            if stop_idx == len(l_values) - 1:
                break
        return result, "ok"


def _parse_tuple_int_value(value, type):
    """Parse an int tuple.

    Parameters
    ----------
    value : tuple(int)
        Tuple that should be parsed.
    type : str
        real pos int: > 0
        real neg int: < 0
        pos int: >= 0
        neg int: <= 0

    Returns
    -------
    tuple(tuple(int), str)
        Returns the tuple(int) and a string with 'ok' if no error occurs. In
        case of an error a tuple('error', error message) is returned.

    """
    if len(value) < 4:
        return ("error", "Tuple must be at least '(x, )'")
    if value[0] != "(" or value[-1] != ")":
        return ("error", "Tuple must be at least '(x, )'")
    t_values = value[1:-1]
    t_values = t_values.split(",")
    if len(t_values) < 2:
        return ("error", "Tuple must be at least '(x, )'")
    result = []
    for i in range(len(t_values)):
        t_val = t_values[i]
        if t_val == "" or t_val == " ":
            continue
        int_type = type[6:]
        value, msg = _parse_int_value(t_val, int_type)
        if value == "error":
            return (value, msg)
        result.append(value)
    result = tuple(result)
    return (result, "ok")


def parse_tuple_int_value(usr_cmds, i, type):
    """Parse a tuple with int values from a user command list.

    Parameters
    ----------
    usr_cmds : list(str)
        List where int values are stored.
    i : int
        The i-th value should be the name of the tuple. The (i+1)-th
        value should be the tuple itself.
    type : str
        real pos int: > 0
        real neg int: < 0
        pos int: >= 0
        neg int: <= 0

    Returns
    -------
    tuple(tuple(int), str)
        Returns the tuple(int) and a string with 'ok' if no error occurs. In
        case of an error a tuple('error', error message) is returned.

    """
    try:
        value = usr_cmds[i + 1]
    except IndexError:
        return ("error", "No value")
    return _parse_tuple_int_value(value, type)