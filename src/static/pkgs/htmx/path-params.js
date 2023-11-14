htmx.defineExtension('path-params', {
    onEvent: function (name, evt) {
        if (name === "htmx:configRequest") {
            evt.detail.path = evt.detail.path.replace(/{([A-Za-z0-9_]+)}/g, function (_, param) {
                let val = evt.detail.parameters[param];
                delete evt.detail.parameters[param];
                return val;
            })
        }
    }
});
