from finance_ml.config.configuration import ConfigurationManager
from finance_ml.components.model_evaluation import ModelEvaluation
from finance_ml import logger

STAGE_NAME= "Model Evaluation stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config_manager = ConfigurationManager()
            evaluation_config = config_manager.get_model_evaluation_config()
            
            model_evaluation = ModelEvaluation(config = evaluation_config)
            model_evaluation.evaluate()

            logger.info (f"{STAGE_NAME} Completed successfully")

        except Exception as e:
            logger.exception(e)
            raise e
        
if __name__ == '__main__':
    try:
        logger.info(f'>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<')
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f'>>>>>>>>> stage {STAGE_NAME} Completed <<<<<<<<<\n\nx==========x')
    except Exception as e:
        logger.exception(e)
        raise e