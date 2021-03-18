// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import * as vscode from 'vscode';
import * as tf from '@tensorflow/tfjs-node';
import { TFSavedModel } from '@tensorflow/tfjs-node/dist/saved_model';

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

let model: TFSavedModel | undefined;

function runModel(content: string): { languageId: ModelLangToLangId; confidence: number } {
	if (!content) {
		return {
			languageId: ModelLangToLangId.bat,
			confidence: 0,
		};
	}

	// call out to the model
	const predicted = model!.predict(tf.tensor([content]));

	const langs: Array<keyof typeof ModelLangToLangId> = (predicted as tf.Tensor<tf.Rank>[])[0].dataSync() as any;
	const probabilities = (predicted as tf.Tensor<tf.Rank>[])[1].dataSync() as Float32Array;

	let maxIndex = 0;
	for (let i = 0; i < probabilities.length; i++) {
		if (probabilities[i] > probabilities[maxIndex]) {
			maxIndex = i;
		}
	}

	return {
		languageId: ModelLangToLangId[langs[maxIndex]],
		confidence: probabilities[maxIndex]
	};
}

export async function activate(context: vscode.ExtensionContext) {
	try {
		model = await tf.node.loadSavedModel(context.asAbsolutePath('model'), ['serve'], 'serving_default');
	} catch (e) {
		vscode.window.showErrorMessage(`Unable to load ML model: ${e}`);
	}

	context.subscriptions.push(vscode.workspace.onDidChangeTextDocument(async (e) => {
		if (e.document.isUntitled && e.document.languageId === 'plaintext') {
			const result = runModel(e.document.getText());
			if (result.confidence >= 0.85) {
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
