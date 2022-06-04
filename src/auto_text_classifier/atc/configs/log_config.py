import os
import logging
import logging.handlers

S_LOG_FORMAT = "[%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s] %(message)s"
S_LOG_SUFFIX = "%Y-%m-%d_%H-%M-%S.log"

def init_logger(s_log_local_path, b_log_debug=False, mode="day"):
    s_log_name = os.path.basename(s_log_local_path)
    
    # 一天一个日志
    logger_handler = logging.handlers.TimedRotatingFileHandler(s_log_local_path, 'midnight', 1, 0, encoding="utf-8")
    logger_handler.suffix = S_LOG_SUFFIX
    logger_handler.setFormatter(logging.Formatter(S_LOG_FORMAT))

    run_logger = logging.getLogger(s_log_name)
    run_logger.setLevel(logging.INFO)
    run_logger.addHandler(logger_handler)

    if b_log_debug:
        consle_handler = logging.StreamHandler()
        consle_handler.setFormatter(logging.Formatter(S_LOG_FORMAT))
        run_logger.addHandler(consle_handler)

    return run_logger