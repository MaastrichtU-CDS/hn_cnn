def get_parameter(parameter, parameters, default_parameters):
    """ Get parameter from the environmnet variables, otherwise
        use the default value provided.
    """
    return parameters[parameter] if parameter in parameters \
        else default_parameters[parameter]

def update_parameters(parameters, default_parameters):
    """ Get parameter from the environmnet variables, otherwise
        use the default value provided.
    """
    update_parameters = default_parameters
    for parameter, value in parameters.items():
        update_parameters[parameter] = value
    return update_parameters
