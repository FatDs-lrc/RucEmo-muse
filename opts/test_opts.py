from .base_opts import BaseOptions

class TestOptions(BaseOptions):
    """This class includes testing options.
    
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--target', type=str, default='arousal', help='arousal, valence')
        parser.add_argument('--submit_dir', type=str, default='submit', help='submit save dir')
        parser.add_argument('--template_dir', type=str, default='submit/template/wild/label_segments/', help='submit save dir')
        parser.add_argument('--test_checkpoints', type=str, help='Use which models as final submition(ensemble)')
        parser.add_argument('--write_sub_results', default=False, action='store_true', help='whether to store result of all checkpoints')
        self.isTrain = False
        return parser