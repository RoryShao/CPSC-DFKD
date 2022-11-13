import logging
from logging import handlers


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, filename, level='info', when='D', backCount=3, fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8')

        th.setFormatter(format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)


if __name__ == '__main__':
    log = Logger('info.log', level='info')
    info = 'Namespace(batch_size=128, ckpt=\'checkpoint/teacher/aircraft-resnet34-224x.pt\', data_root=\'/data/rrshao/datasets/aircraft\', dataset=\'aircraft\', download=False, epoch_itrs=50, epochs=600, log_interval=10, lr_G=0.001, lr_S=0.1, model=\'resent18\', momentum=0.9, no_cuda=False, nz=256, operator=\'DFAD\', scheduler=True, seed=1, step_size=100, temp=0.07, test_batch_size=128, test_only=False, weight_decay=0.0005)'
    for i in range(20):
        log.logger.info("epoch "+str(i)+": "+info)