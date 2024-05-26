""""Make batch predictions on new data"""

from metaflow import FlowSpec, step, Flow, current
from pathlib import Path


class NLPPredictionFlow(FlowSpec):
    """Make batch predictions on new data"""

    outputs_dir = Path('../data/05_outputs')

    @step
    def start(self):
        """Get the latest deployment candidate that is from a successfull run"""
        from utils.runs import get_latest_successful_run
        print('Loading data of NLPFlow last run')
        self.deploy_run = get_latest_successful_run(
            'NLPFlow', 'deployment_candidate')
        self.next(self.end)

    @step
    def end(self):
        "Make predictions"
        from models.bow import NbowModel
        import pandas as pd
        import pyarrow as pa

        print('Loading data to predict')
        new_reviews = pd.read_parquet(
            self.outputs_dir / 'predict.parquet')['review']

        print('Loading model and making predictions')
        model = NbowModel.from_dict(
            self.deploy_run.data.model_dict)
        predictions = model.predict(new_reviews)
        
        print(f'Writing predictions to parquet: {predictions.shape[0]} rows')
        table = pa.table({"data": predictions})
        filename = self.outputs_dir / "sentiment_predictions.parquet"
        pa.parquet.write_table(table, filename)


if __name__ == '__main__':
    NLPPredictionFlow()
