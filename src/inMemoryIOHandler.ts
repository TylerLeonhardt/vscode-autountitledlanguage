import { io } from "@tensorflow/tfjs";
import * as path from 'path';
import * as fs from 'fs';

export class InMemoryIOHandler implements io.IOHandler {
	private static modelPath = path.join(__dirname, '..', 'model', 'model.json');
	private static weightsPath = path.join(__dirname, '..', 'model', 'group1-shard1of1.bin');

	async load(): Promise<io.ModelArtifacts> {
		let modelJSON: io.ModelJSON;
		try {
			modelJSON = require(InMemoryIOHandler.modelPath);
		} catch (e) {
			const message = `Failed to parse model JSON of response from ${InMemoryIOHandler.modelPath}.`;
			throw new Error(message);
		}

		// We do not allow both modelTopology and weightsManifest to be missing.
		const modelTopology = modelJSON.modelTopology;
		const weightsManifest = modelJSON.weightsManifest;
		if (modelTopology === null && weightsManifest === null) {
			throw new Error(
				`The JSON from path ${InMemoryIOHandler.modelPath} contains neither model topology or manifest for weights.`);
		}

		return this.getModelArtifactsForJSON(
			modelJSON, (weightsManifest) => this.loadWeights(weightsManifest));
	}

	private async getModelArtifactsForJSON(
		modelJSON: io.ModelJSON,
		loadWeights: (weightsManifest: io.WeightsManifestConfig) => Promise<[
		  /* weightSpecs */ io.WeightsManifestEntry[], /* weightData */ ArrayBuffer
		]>): Promise<io.ModelArtifacts> {
		const modelArtifacts: io.ModelArtifacts = {
			modelTopology: modelJSON.modelTopology,
			format: modelJSON.format,
			generatedBy: modelJSON.generatedBy,
			convertedBy: modelJSON.convertedBy
		};

		if (modelJSON.trainingConfig !== null) {
			modelArtifacts.trainingConfig = modelJSON.trainingConfig;
		}
		if (modelJSON.weightsManifest !== null) {
			const [weightSpecs, weightData] =
				await loadWeights(modelJSON.weightsManifest);
			modelArtifacts.weightSpecs = weightSpecs;
			modelArtifacts.weightData = weightData;
		}
		if (modelJSON.signature !== null) {
			modelArtifacts.signature = modelJSON.signature;
		}
		if (modelJSON.userDefinedMetadata !== null) {
			modelArtifacts.userDefinedMetadata = modelJSON.userDefinedMetadata;
		}
		if (modelJSON.modelInitializer !== null) {
			modelArtifacts.modelInitializer = modelJSON.modelInitializer;
		}

		return modelArtifacts;
	}

	private async loadWeights(weightsManifest: io.WeightsManifestConfig): Promise<[io.WeightsManifestEntry[], ArrayBuffer]> {
		const weightSpecs = [];
		for (const entry of weightsManifest) {
			weightSpecs.push(...entry.weights);
		}

		return [weightSpecs, fs.readFileSync(InMemoryIOHandler.weightsPath).buffer];
	}
}
