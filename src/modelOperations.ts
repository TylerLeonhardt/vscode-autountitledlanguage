import * as tf from '@tensorflow/tfjs';
import { InMemoryIOHandler } from './inMemoryIOHandler';
import { ModelLangToLangId, ModelResult } from './types';
// Adds the CPU backend to the global backend registry.
import '@tensorflow/tfjs-backend-cpu';

export class ModelOperations {
    static modelCache: tf.GraphModel | undefined;

    public async loadModel() {
        if (ModelOperations.modelCache) {
            return;
        }

        await tf.setBackend('cpu');
        ModelOperations.modelCache = await tf.loadGraphModel(new InMemoryIOHandler());
    }

    public async runModel(content: string): Promise<Array<ModelResult>> {
        if (!content) {
            return [];
        }
    
        // call out to the model
        const predicted = await ModelOperations.modelCache!.executeAsync(tf.tensor([content]));
    
        const probabilities = (predicted as tf.Tensor<tf.Rank>[])[0].dataSync() as Float32Array;
        const langs: Array<keyof typeof ModelLangToLangId> = (predicted as tf.Tensor<tf.Rank>[])[1].dataSync() as any;
    
        const objs: Array<ModelResult> = [];
        for (let i = 0; i < langs.length; i++) {
            objs.push({
                languageId: ModelLangToLangId[langs[i]],
                confidence: probabilities[i],
            });
        }
    
        let maxIndex = 0;
        for (let i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > probabilities[maxIndex]) {
                maxIndex = i;
            }
        }
    
        return objs.sort((a, b) => {
            return b.confidence - a.confidence;
        });
    }

    public dispose() {
        ModelOperations.modelCache?.dispose();
    }
}
