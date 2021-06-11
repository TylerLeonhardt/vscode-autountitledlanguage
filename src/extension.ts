// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import * as vscode from 'vscode';
// Import @tensorflow/tfjs-core
import * as tf from '@tensorflow/tfjs';
// Adds the CPU backend to the global backend registry.
import '@tensorflow/tfjs-backend-cpu';
import * as path from 'path';

enum ModelLangToLangId {
	bat = 'bat',
	cmd = 'bat',
	btm = 'bat',
	c = 'c',
	cs = 'csharp',
	cpp = 'cpp',
	cc = 'cpp',
	coffee = 'coffeescript',
	litcoffee = 'coffeescript',
	css = 'css',
	erl = 'erlang',
	hrl = 'erlang',
	go = 'go',
	hs = 'haskell',
	lhs = 'haskell',
	html = 'html',
	java = 'java',
	js = 'javascript',
	es6 = 'javascript',
	ipynb = 'jupyter',
	lua = 'lua',
	md = 'markdown',
	matlab = 'matlab',
	m = 'objective-c',
	mm = 'objective-c',
	pl = 'perl',
	pm = 'perl',
	php = 'php',
	ps1 = 'powershell',
	py = 'python',
	r = 'r',
	rdata = 'r',
	rds = 'r',
	rda = 'r',
	rb = 'ruby',
	rs = 'rust',
	scala = 'scala',
	sh = 'shellscript',
	sql = 'sql',
	swift = 'swift',
	tex = 'tex',
	ts = 'typescript',
	tsx = 'typescriptreact'
};

interface ModelResult {
	languageId: ModelLangToLangId;
	confidence: number;
}

let modelCache: tf.GraphModel | undefined;

async function runModel(content: string): Promise<Array<ModelResult>> {
	if (!content) {
		return [];
	}

	// call out to the model
	const predicted = await modelCache!.executeAsync(tf.tensor([content]));

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

async function loadModel() {
	await tf.setBackend('cpu');
	try {
		modelCache = await tf.loadGraphModel('https://storage.googleapis.com/tfjs-testing/guesslang-demo/model.json');
	} catch (e) {
		vscode.window.showErrorMessage(`Unable to load ML model: ${e}`);
	}
}

export async function activate(context: vscode.ExtensionContext) {
	await loadModel();

	context.subscriptions.push(vscode.workspace.onDidOpenTextDocument(async (e) => {
		if (!modelCache) {
			await loadModel();
		}
	}));

	context.subscriptions.push(vscode.workspace.onDidCloseTextDocument((e) => {
		if (!vscode.workspace.textDocuments.some(t => t.isUntitled && t.languageId === 'plaintext')) {
			if(modelCache) {
				modelCache.dispose();
				modelCache = undefined;
			}
		}
	}));

	context.subscriptions.push(vscode.workspace.onDidChangeTextDocument(async (e) => {
		if (e.document.isUntitled && e.document.languageId === 'plaintext') {
			const modelResults = await runModel(e.document.getText());
			if (!modelResults) {
				return;
			}

			const result = modelResults[0];

			// For ts/js and c/cpp we "add" the confidence of the other language so ensure better results
			switch (result.languageId) {
				case ModelLangToLangId.ts:
					if (modelResults[1].languageId === ModelLangToLangId.js) {
						result.confidence += modelResults[1].confidence;
					}
					break;
				case ModelLangToLangId.js:
					if (modelResults[1].languageId === ModelLangToLangId.ts) {
						result.confidence += modelResults[1].confidence;
					}
					break;
				case ModelLangToLangId.c:
					if (modelResults[1].languageId === ModelLangToLangId.cpp) {
						result.confidence += modelResults[1].confidence;
					}
					break;
				case ModelLangToLangId.cpp:
					if (modelResults[1].languageId === ModelLangToLangId.c) {
						result.confidence += modelResults[1].confidence;
					}
					break;
				default:
					break;
			}

			if (result.confidence >= 0.6) {
				const langIds = await vscode.languages.getLanguages();
				if (!langIds.some(l => l === result.languageId)) {
					// The language isn't supported in VS Code
					return;
				}

				await vscode.languages.setTextDocumentLanguage(e.document, result.languageId);
			}
		}
	}));
}

export function deactivate() { }
