// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
const vscode = require('vscode');
const fetch = (...args) => import('node-fetch').then(({default: fetch}) => fetch(...args));
const uuidv1 = require('uuid').v1;
const path = require('path');
const fs = require('fs');

let global_context; 
// Only to be used for cleaning up after the extension deactivates
let panel_cleanup;

const sleep = (delay) => new Promise((resolve) => setTimeout(resolve, delay));


// =============
// Communication
// =============

/**
 * Sends a post request to a server specified in the configuration.
 * @param data Will be sent as payload
 * @return Response is returned as an output.
 */
const postData = async (data = {}) => {
    let config = vscode.workspace.getConfiguration('codeimprove');
    let url = `http://${config.get('serverAddress')}`;
    const options = {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
    };

    try {
        const response = await fetch(url, options);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const jsonResponse = await response.json();
        console.debug("Recived:", jsonResponse);

        return jsonResponse;
    }
    catch (error) {
        console.error('Request failed', error);
        return null;
    }
}

/**
 * Request alternative names for a given variable.
 * @param task Special task object
 * @return Special object as per communication convention.
 */
const requestRename = async (task) => {
    const myUUID = uuidv1();
    data =
        {
			uuid: `${myUUID}`,
			requests: [
				{
					id: 1,
					tasks: [
						{
							task: "rename",
							symbol: `${task.rename_variable.name}`
						}
					],
					snippet: `${task.code}`
				}
			]
		};
    console.debug(`Sending request for rename.`, data);

    const response = await postData(data);
    if (!response) {
        console.warn(`Empty reponse for rename of ${word}`);
    }

    return response.response[0].result;
}

/**
 * Request Bug fix for a given function.
 * @param task Special task object
 * @return Special object as per communication convention.
 */
const requestErrorFix = async (task) => {
    const myUUID = uuidv1();
    data =
        {
			uuid: `${myUUID}`,
			requests: [
				{
					id: 1,
					tasks: [
						{
							task: "error"
						}
					],
					snippet: `${task.code}`
				}
			]
		};
    console.debug(`Sending request for errorfix.`, data);

    const response = await postData(data);
    if (!response) {
        console.warn(`Empty response for rename of ${word}`);
        return null;
    }

    return response.response[0].result;
}

/**
 * Request comments for a given function.
 * @param task Special task object
 * @return Special object as per communication convention.
 */
const requestComments = async (task) => {
    const myUUID = uuidv1();
    const style = global_context.globalState.get("docstring_style", "NA");
    data =
        {
			uuid: `${myUUID}`,
			requests: [
				{
					id: 1,
					tasks: [
						{
							task: "comment",
                            type: style,
						}
					],
					snippet: task.code
				}
			]
		};
    console.debug(`Sending request for rename .`, data);

    const response = await postData(data);
    if (!response) {
        console.warn(`Empty reponse for rename of ${word}`);
    }

    return response.response[0].result;
}


// =========
// Code Lens
// =========

/**
 * Find function symbols from a tree off symbols, runs recursively.
 * @param symbols a list of VSCode symbols.
 * @return Returns a list of all function and method symbols
 */
const findFunctionSymbol = (symbols) => {
    let functions = []
    for (let symbol of symbols) {
        if (symbol.kind === vscode.SymbolKind.Method || symbol.kind === vscode.SymbolKind.Function) {
            functions.push(symbol)
        } 
        else if (symbol.children && symbol.children.length > 0) {
            functions = [ ...functions, ...findFunctionSymbol(symbol.children)];
        }
    }
    return functions;
}

/**
 * Checks for variable declaration in the highlighted area of a document
 * @param symbols a list of VSCode symbols.
 * @param editor a reference to the current editor object
 * @return null if no variable was found or a wrapper object containing 
 * the variable as symbol and its parent as parent
 */
const findVariableInSelection = (symbols, editor) => {
    const selection_range = editor.selection.active
    if (!selection_range) {
        return null;
    }

    // We are looking for both the variable declaration that intersects with 
    // the selection and its parent. We only support renaming variables that 
    // are inside functions or methods so we need to exclude all others.
    const recursive_search = (symbols, selection_range, root=true) => {
        for (let symbol of symbols) {
            const symbolRange = symbol.location.range
            if (!symbolRange.isEqual(selection_range) && !symbolRange.contains(selection_range)) {
                continue;
            }
            if (symbol.kind !== vscode.SymbolKind.Variable) {
                if (symbol.children) {
                    const result = recursive_search(symbol.children, selection_range, false);
                    if (result) {
                        if(!result.parent) {
                            result.parent = symbol
                        }
                        if (result.parent.kind !== vscode.SymbolKind.Function && result.parent.kind !== vscode.SymbolKind.Method) {
                            return null;
                        }
                        return result;
                    }
                    continue;
                }
            }
            // Do not include variables in the root scope
            if (root) {
                continue;
            }

            return {symbol, parent: null};
        }
        return null;
    }

    return recursive_search(symbols, selection_range);
}


/**
 * CodeLens implementation for marking up relevant parts of code with
 * buttons for triggering refactoring actions this extension offers
 */
class ImproveLens {
    constructor() {
        this._onDidChangeCodeLenses = new vscode.EventEmitter();
        this.onDidChangeCodeLenses = this._onDidChangeCodeLenses.event;
    }

    async provideCodeLenses(document, _) {
        const editor = vscode.window.visibleTextEditors.find(editor => editor.document === document);

        let symbols = null
        while (true) {
            symbols = await vscode.commands.executeCommand('vscode.executeDocumentSymbolProvider', editor.document.uri);
            // Symbol search might fail if the editor isn't fully loaded yet.
            if (!symbols) {
                await sleep(500);
                console.debug("Retrying symbol scanning...");
            }
            else {
                break;
            }
        }

        let lenses = []
        const functions = findFunctionSymbol(symbols);
        const variableResult = findVariableInSelection(symbols, editor);
        if (variableResult) {
            const {symbol: variableSymbol, parent: variableParent} = variableResult;
            const renameTask = { name: "rename", fn: requestRename, rename_variable: variableSymbol};
            lenses.push(new vscode.CodeLens(variableSymbol.location.range,{
                title: `Rename ${variableSymbol.name}`,
                command: "codeimprove.openCommentsPanel",
                arguments: [variableParent.range, editor, renameTask],
            }));
        }

        for (let fn of functions) {
            const line = fn.range.start.line;
            const range = new vscode.Range(line, 0, line, 0);

            const commentTask = { name: "comment", fn: requestComments};
            lenses.push(new vscode.CodeLens(range,{
                title: "Suggest comments",
                command: "codeimprove.openCommentsPanel",
                arguments: [fn.range, editor, commentTask],
            }));

            const errorTask = { name: "error", fn: requestErrorFix };
            lenses.push(new vscode.CodeLens(range,{
                title: "Fix error",
                command: "codeimprove.openCommentsPanel",
                arguments: [fn.range, editor, errorTask],
            }));
        }

        return lenses;
    };

    refresh() {
        // Trigger the event to refresh CodeLenses
        this._onDidChangeCodeLenses.fire();
    }
    
}


// =========
// Rendering
// =========

/**
 * Fills a templated HTML file for the loading page.
 * @param type Type of task
 * @param panel Reference to the panel this html is eventually going 
 * to be displayed in
 * @return The HTML as a string
 */
const renderLoadingHTML = (type, panel) => {
    const templatePath = path.join(global_context.extensionPath, 'webviews', 'loading.html');
    let htmlContent = fs.readFileSync(templatePath, 'utf8');
    const vscodeCssUri = panel.webview.asWebviewUri(vscode.Uri.file(
        path.join(global_context.extensionPath, 'webviews', 'vscode.css')
    ));

    htmlContent = htmlContent.replace(/\{\{vscodeCssUri\}\}/g, vscodeCssUri);
    htmlContent = htmlContent.replace('{{type}}', type);
    return htmlContent;
}

/**
 * Basic function for escaping potentially dangerous characters from 
 * user-generated input
 * @param unsafe The untrusted HTML
 * @return A safe HTML
 */
const escapeHtml = (unsafe) => {
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
}

/**
 * Fills a templated HTML file for the main data display page.
 * @param type Special task object
 * @param data The payload that will be displayed in the code view.
 * @param panel Reference to the panel this html is eventually going 
 * to be displayed in
 * @return The HTML as a string
 */
const renderDataHTML = (task, data, panel) => {
    let splitCode = task.code.split("\n");
    let visual_code_split = [...splitCode]

    const changed_lines = [];
    if (task.name === "comment") {

        let inserted = 0;

        for(const key of Object.keys(data).sort((a, b) => a - b)) {
            let index; 
            if (key == "0") {
                console.log("docstasdfasdfasd ", data[key]);
                const indent = splitCode[1].match(/^\s*/)[0]
                const docstr = data[key].replace('\n', `\n${indent}`);
                const lines = docstr.split('\n').length 
                splitCode[1] = `${indent}"""${docstr.trim()}\n${indent}"""\n${splitCode[1]}`;
                changed_lines.push(2);
                inserted += lines;
                continue;
            }
            else {
                index = Number(key)-1;
            }
            const indent = splitCode[index].match(/^\s*/)[0]
            splitCode[index] = `${indent}# ${data[key].trim()}\n${splitCode[index]}`;
            inserted += 1;
            changed_lines.push(index+inserted+1);
        }


        if (Object.keys(data).length <= 0) {
            return null;
        }
        visual_code_split = splitCode
    }
    else if (task.name === "error") {
        const regex = /\$(\d+)\$(\s*.*?)(?=\$\d+\$|$)/g;
        let extra_lines = 0;
        let found_any = false;
        while ((match = regex.exec(data)) !== null) {
            found_any = true;
            const index = Number(match[1])-1

            if (splitCode[index] == match[2] && match[2]) {
                continue;
            }

            let before_part = "";
            let after_part = "";
            let after_change_comment = "New";
            let before_change_comment = "Before";
            let separator = "\n";

            // The line is present in the original, therefore its not 
            // an addition but a replacement
            if (splitCode[index]) {
                after_change_comment = "After";
            }

            // Line exists in the original and is not being deleted,
            // that means that line count will be offset by one.
            if (splitCode[index] && match[2]) {
                extra_lines += 1
            }

            // Patch removes the line, we do not want to display the patch line.
            if (!match[2]) {
                before_change_comment = "Removed";
                separator = "";
                extra_lines += 1
            }
            // When patch does not remove the line we want to display the patch.
            else {
                after_part = `${match[2]}\t# ${after_change_comment}`
                // Multiple highlights have different color
                for (let i = 0; i < 5; i++) {
                    changed_lines.push(index+extra_lines+1);
                }
            }

            // When the line that is being patched is present in the original we
            // want to display its before section and highlight it.
            if (splitCode[index]) {
                before_part = `${splitCode[index]}\t# ${before_change_comment}`;
                changed_lines.push(index+extra_lines);
            }
            else if (!splitCode[index] && !match[2]) {
                before_part = `# ${before_change_comment}`;
                changed_lines.push(index+extra_lines);
            }
            else { 
                separator = "";
            }

            // I don't know
            if (!match[2]) {
                extra_lines -= 1
            }

            splitCode[index] = match[2]
            visual_code_split[index] = `${before_part}${separator}${after_part}`
        }
        if (!found_any) {
            return null;
        }
    }

    const mergedCode = splitCode.join("\n");
    const mergedVisualCode = escapeHtml(visual_code_split.join("\n"));

    // Get paths to resources on disk
    const prismJsUri = panel.webview.asWebviewUri(vscode.Uri.file(
        path.join(global_context.extensionPath, 'webviews', 'prism.min.js')
    ));
    const prismCssUri = panel.webview.asWebviewUri(vscode.Uri.file(
        path.join(global_context.extensionPath, 'webviews', 'prism.min.css')
    ));
    const vscodeCssUri = panel.webview.asWebviewUri(vscode.Uri.file(
        path.join(global_context.extensionPath, 'webviews', 'vscode.css')
    ));

    const css = path.join(global_context.extensionPath, 'webviews', 'vscode.css');
    const templatePath = path.join(global_context.extensionPath, 'webviews', 'display.html');
    let htmlContent = fs.readFileSync(templatePath, 'utf8');

    const comment_style = global_context.globalState.get("docstring_style", "NA"); 

    // Filling the template
    htmlContent = htmlContent.replace(/\{\{type\}\}/g, task.name);
    htmlContent = htmlContent.replace(/\{\{css\}\}/g, css);
    htmlContent = htmlContent.replace(/\{\{docstring_style\}\}/g, comment_style);
    htmlContent = htmlContent.replace(/\{\{prismCssUri\}\}/g, prismCssUri);
    htmlContent = htmlContent.replace(/\{\{vscodeCssUri\}\}/g, vscodeCssUri);
    htmlContent = htmlContent.replace(/\{\{prismJsUri\}\}/g, prismJsUri);
    htmlContent = htmlContent.replace(/\{\{highlights\}\}/g, changed_lines.join(','));
    htmlContent = htmlContent.replace(/\{\{data\}\}/g, mergedVisualCode);

    if (task.name === "rename") {
        htmlContent = htmlContent.replace(/\{\{rename_option_1\}\}/g, escapeHtml(data[0].symbol));
        htmlContent = htmlContent.replace(/\{\{rename_option_2\}\}/g, escapeHtml(data[1].symbol));
        htmlContent = htmlContent.replace(/\{\{rename_option_3\}\}/g, escapeHtml(data[2].symbol));
        htmlContent = htmlContent.replace(/\{\{rename_variable\}\}/g, escapeHtml(task.rename_variable.name));
    }

    return {htmlContent, mergedCode};
}


// =====
// Panel
// =====

/**
 * Function controlling the panel with suggestions.
 * @param range The range of the relevant function in the source
 * @param editor The reference to the editor.
 * @param task A special object
 */
const openPanel = (range, editor, task) => {
    let fixedCode;
    const panel = vscode.window.createWebviewPanel(
        'inspector',
        'CodeImprove',
        vscode.ViewColumn.Beside,
        {
            enableScripts: true,
            sandboxOptions: {
                permissions: ['allow-scripts']
            },
        }
    );

    // Saving panel aside for cleaning up during deactivation
    panel_cleanup = panel.dispose;

    // Set the webview's content
    panel.reveal(panel.viewColumn, false);

    // Close panel when focus is lost.
    panel.onDidChangeViewState(e => {
        if (!e.webviewPanel.active) {
            panel.dispose();
        }
    });

    const acceptChange = (fixedCode, range, editor) => {
        vscode.commands.executeCommand('codeimprove.replaceText', fixedCode, range, editor);
        panel.dispose();
    }

    panel.webview.onDidReceiveMessage(message => {
        switch (message.command) {
            case 'accept':
                acceptChange(fixedCode, range, editor);
                break;
            case 'close':
                panel.dispose();
                break;
            case 'change_docstring_style':
                panel.dispose();
                global_context.globalState.update("docstring_style", message.style).then(
                    () => { openPanel(range, editor, task); }
                );
                break;
            case 'choose_name':
                panel.dispose();
                acceptChange(response_data[message.choice_index].code, range, editor); 
                break;
            default:
                console.warn(`Unhandled command ${message.command}`);
        }
    });

    let response_data;
    panel.webview.html = renderLoadingHTML(task.name, panel);
    task.code = editor.document.getText(range);
    task.fn(task).then(
        data => {
            response_data = data;
            const output = renderDataHTML(task, data, panel)
            if (!output) {
                panel.dispose();
                vscode.window.showInformationMessage('No suggestions for this code.');
                return;
            }
            panel.webview.html = output.htmlContent;
            fixedCode = output.mergedCode;
        }
    ).catch(
        error => {
            panel.dispose();
            console.error(error);
            vscode.window.showErrorMessage('Error generating improvements, check your connection and try again later.');
        }
    );
}

const replaceInRange = (text, range, editor) => {
    editor.edit(editBuilder => {
        editBuilder.replace(range, text);
    });
}

/**
 * @param {vscode.ExtensionContext} context
 */
const activate = (context) => {
    global_context = context;
    
    context.subscriptions.push(vscode.commands.registerCommand('codeimprove.openCommentsPanel', openPanel));
    context.subscriptions.push(vscode.commands.registerCommand("codeimprove.replaceText", replaceInRange));
    const provider = new ImproveLens();
    const selector = { language: 'python', scheme: 'file' };
    context.subscriptions.push(vscode.languages.registerCodeLensProvider(selector, provider));
    context.subscriptions.push(vscode.window.onDidChangeTextEditorSelection(event => {
    if (event.textEditor.document.languageId === "python") {
        provider.refresh();
    }
    }));

	console.log('CodeImprove extension loaded succesfully. Created by Ondřej Zobal in 2024.');
}

module.exports = {
	activate,
	deactivate: () => {
        try {
            panel_cleanup()
        }
        catch (_) {
            // If the cleanup failed that means that 
            // there was no panel and everything is
            // ok.
        }
    },
}
