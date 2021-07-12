// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import * as vscode from 'vscode';
import { ModelLangToLangId } from './types';
import { ModelOperations } from './modelOperations';

export async function activate(context: vscode.ExtensionContext) {
	const modelOperations = new ModelOperations();
	await modelOperations.loadModel();

	context.subscriptions.push(vscode.workspace.onDidOpenTextDocument(async (e) => {
		await modelOperations.loadModel();
	}));

	context.subscriptions.push(vscode.workspace.onDidCloseTextDocument((e) => {
		if (!vscode.workspace.textDocuments.some(t => t.isUntitled && t.languageId === 'plaintext')) {
			modelOperations.dispose();
		}
	}));

	context.subscriptions.push(vscode.workspace.onDidChangeTextDocument(async (e) => {
		if (e.document.isUntitled && e.document.languageId === 'plaintext') {
			const modelResults = await modelOperations.runModel(e.document.getText());
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
