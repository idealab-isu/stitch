import logging
import torch.distributed as dist
import numpy as np
import torch
import gudhi as gd
import matplotlib.pyplot as plt
from gudhi.wasserstein import wasserstein_distance

logger_initialized = {}

def get_root_logger(log_file=None, log_level=logging.INFO, name='main'):
    """Get root logger and add a keyword filter to it.
    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmdet3d".
    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
        name (str, optional): The name of the root logger, also used as a
            filter keyword. Defaults to 'mmdet3d'.
    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name=name, log_file=log_file, log_level=log_level)
    # add a logging filter
    logging_filter = logging.Filter(name)
    logging_filter.filter = lambda record: record.find(name) != -1

    return logger


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    """Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.
    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.
    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # handle duplicate logs to the console
    # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
    # to the root logger. As logger.propagate is True by default, this root
    # level handler causes logging messages from rank>0 processes to
    # unexpectedly show up on the console, creating much unwanted clutter.
    # To fix this issue, we set the root logger's StreamHandler, if any, to log
    # at the ERROR level.
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True


    return logger


def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.
    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')

def apply_l1_reg(pred_sdf, l1_penalty):

    l1_loss = torch.sum(torch.abs(pred_sdf))
    
    return l1_loss * l1_penalty

def apply_l2_reg(pred_sdf, l2_penalty):

    l2_loss = torch.sum(pred_sdf ** 2)

    return l2_loss * l2_penalty


def topological_loss_2(diag2):
    """
    Computes loss in eqn 14 of "Topology-controllable Implicit Surface Reconstruction Based on
    Persistent Homology"
    """
    # minimize birth times of diag2
    temp_loss = 0
    final_loss = []
    for dim in range(len(diag2)):

        # print(f'dimension: {dim}, top_loss diag2_length: {len(diag2)}')
        # print(f'len(diag2[dimension]): {len(diag2[dim])}')
        if len(diag2[dim]) != 0:
            diag2 = torch.tensor(diag2[dim])
       
            # delete index with death time = inf
            if diag2.shape[0] > 2:
                diag2 = diag2[diag2[:, 1] != np.inf]
                term_1 = torch.sum(diag2[:, 0])
                term_2 = torch.sum(diag2[:, 1] - diag2[:, 0])
                # temp_loss = term_1 + term_2
                temp_loss = term_1
                final_loss.append(temp_loss)
                temp_loss = 0
            else:
                final_loss.append(0)
        else:
            final_loss.append(0)

    # return term_1 + term_2
    return sum(final_loss)


def wasserstein_metric(diag1, diag2, max_edge_length):
    loss = gd.wasserstein.wasserstein_distance(diag1, diag2, order=2, internal_p=2, keep_essential_parts=False, enable_autodiff=True)
    return loss

def feature_extract(input_barcodes, threshold):
    # Identify noise
    noise_pairs = []
    significant_pairs = []
    for pair in input_barcodes:
        if pair[1] == np.inf:
            pass
        else:
            birth, death = pair[0], pair[1]
            if death - birth < threshold:
                # if death - birth > threshold:
                noise_pairs.append(pair)
            else:
                significant_pairs.append(pair)

    return significant_pairs, noise_pairs

def topological_loss_1(diag2, max_edge_length):
    """
    Computes loss in eqn 15 of "Topology-controllable Implicit Surface Reconstruction Based on
    Persistent Homology"
    """
    # loss = gd.wasserstein.wasserstein_distance(diag1, diag2, order=2, internal_p=2, keep_essential_parts=False, enable_autodiff=True)

    # go through diag1 and find the the most occuring death time and store it in a list. store the remaining death times in another list
    # count the number of occurences of the most occuring death time and store it in a list. store the remaining death times in another list

    # persistent_features_pc = []
    # unique_death_times_pc, unique_pc_counts = np.unique(diag1[:, 1], return_counts=True)
    # unique_pc_dict = dict(zip(unique_death_times_pc, unique_pc_counts))

    
    dimensional_loss = 0
    final_loss = []
    # print(f'top_loss diag2_length: {len(diag2)}')
    for dimension in range(len(diag2)):
        # print(f'len(diag2[{dimension}]): {len(diag2[dimension])}')
        
        
        # diag1_significant_pairs, diag1_noise_pairs = feature_extract(
        #     diag1, max_edge_length
        # )
        diag2_significant_pairs, diag2_noise_pairs = feature_extract(diag2[dimension], max_edge_length)

        diag2_significant_pairs = np.asarray(diag2_significant_pairs)
        diag2_noise_pairs = np.asarray(diag2_noise_pairs)


        # First summation term
        # first_term = -sum(d - b for b, d in most_persistent)
        first_term = -1 * sum(d - b for b, d in diag2_significant_pairs)
        # first_term = -1 * sum(b - d for b, d in diag2_significant_pairs)  # flipped to b-d to not have negative values
        
        # Second summation term
        # second_term = sum(d - b for b, d in noise_features)
        second_term = sum(d - b for b, d in diag2_noise_pairs)
        # print("second_term: ", second_term)

        # Dimensional loss
        # dimensional_loss = first_term + second_term
        dimensional_loss = first_term
        final_loss.append(dimensional_loss)

    # Total loss
    # sum of all losses in final_loss
    # final_loss = sum(final_loss)
    # loss = first_term + second_term
    return sum(final_loss)

def get_features(features):
    noise_data = []
    significant_data = []
    for pair in features:
        if pair[1] == np.inf:
            pass
        else:
            birth, death = pair[0], pair[1]
            # non_same_zero_dim_barcodes = zero_dim_barcodes[zero_dim_barcodes[:,0] != zero_dim_barcodes[:,1]]
            if birth == death:
                noise_data.append(pair)
            else:
                significant_data.append(pair)

    return significant_data, noise_data

# def feature_extract(input_barcodes, threshold):
#     # Identify noise
#     noise_pairs = []
#     significant_pairs = []
#     for pair in input_barcodes:
#         if pair[1] == np.inf:
#             pass
#         else:
#             birth, death = pair[0], pair[1]
#             if death - birth < threshold:
#                 # if death - birth > threshold:
#                 noise_pairs.append(pair)
#             else:
#                 significant_pairs.append(pair)

#     return significant_pairs, noise_pairs

def new_top_loss1(diag2):
    """
    Computes loss in eqn 15 of "Topology-controllable Implicit Surface Reconstruction Based on
    Persistent Homology"
    """
 
    dimensional_loss = 0
    final_loss = []
    for dimension in range(len(diag2)):
        diag2_significant_pairs, diag2_noise_pairs = get_features(diag2[dimension])
        # print('diag2_significant features: ', diag2_significant_pairs)
        # print('diag2_noise features: ', diag2_noise_pairs)

        diag2_significant_pairs = np.asarray(diag2_significant_pairs)
        diag2_noise_pairs = np.asarray(diag2_noise_pairs)


        # First summation term
        first_term = -1 * sum(d - b for b, d in diag2_significant_pairs)
        
        # Second summation term
        # second_term = sum(d - b for b, d in diag2_noise_pairs)

        # Dimensional loss
        # dimensional_loss = first_term + second_term
        dimensional_loss = first_term
        final_loss.append(dimensional_loss)

    return sum(final_loss)

def new_top_loss2(diag2):
    """
    Computes loss in eqn 14 of "Topology-controllable Implicit Surface Reconstruction Based on
    Persistent Homology"
    """
    # minimize birth times of diag2
    # temp_loss = 0
    final_loss = []

    for dimension in range(len(diag2)):
        diag2_significant_pairs, diag2_noise_pairs = get_features(diag2[dimension])
        diag2_significant_pairs = np.asarray(diag2_significant_pairs)
        diag2_noise_pairs = np.asarray(diag2_noise_pairs)

        # First summation term
        if diag2_significant_pairs.ndim != 1:
            first_term = sum(diag2_significant_pairs[:, 0])
        else:
            print(diag2_significant_pairs)
            first_term = torch.tensor(0.0)
        
        # Second summation term
        second_term = sum(d - b for b, d in diag2_noise_pairs)

        # Dimensional loss
        dimensional_loss = first_term + second_term
        # dimensional_loss = first_term
        final_loss.append(dimensional_loss)

    return sum(final_loss)
    # return final_loss

    # for dimension in range(len(diag2)):
    #     if len(diag2[dimension]) != 0:
    #         diag2 = torch.tensor(diag2[dim])
       
    #         # delete index with death time = inf
    #         if diag2.shape[0] > 2:
    #             diag2 = diag2[diag2[:, 1] != np.inf]
    #             term_1 = torch.sum(diag2[:, 0])
    #             term_2 = torch.sum(diag2[:, 1] - diag2[:, 0])
    #             temp_loss = term_1 + term_2
    #             final_loss.append(temp_loss)
    #             temp_loss = 0
    #         else:
    #             final_loss.append(0)
    #     else:
    #         final_loss.append(0)
    # return sum(final_loss)

def compute_cubical_cmplx(sdf, sdf_res, maxdim):
    sdf_shape = [sdf_res, sdf_res, sdf_res]
    min_sdf = torch.min(sdf)
    mean_sdf = torch.mean(sdf)
    sdf = sdf.reshape(sdf_res,sdf_res,sdf_res)
    skeleton = gd.CubicalComplex(dimensions=sdf_shape, top_dimensional_cells=sdf.detach().cpu().numpy().flatten())
    barcodes = skeleton.persistence(min_persistence=min_sdf)
    domain_all_dim_barcodes = []
    zero_dim_barcodes = []
    for dim in range(maxdim+1):
        dim_barcodes = skeleton.persistence_intervals_in_dimension(dim)
        domain_all_dim_barcodes.append(dim_barcodes)
        if dim == 0:
            zero_dim_barcodes.append(dim_barcodes)

    return domain_all_dim_barcodes, barcodes, zero_dim_barcodes

# def compute_pers_diagram(radius_thresh, pc, maxdim):
#     skeleton = gd.RipsComplex(points=pc.detach().cpu().numpy(), max_edge_length=radius_thresh)
#     simplex_tree = skeleton.create_simplex_tree(max_dimension=maxdim)
#     barcodes = simplex_tree.persistence()
#     zero_dim_barcodes_pd = simplex_tree.persistence_intervals_in_dimension(maxdim)
#     return zero_dim_barcodes_pd, barcodes
