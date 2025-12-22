// ESLint config for the small JS surface area in this repo.
// Focus: enforce high-quality JSDoc blocks for public scripts.

import jsdoc from "eslint-plugin-jsdoc";

export default [
  {
    files: ["scripts/**/*.mjs"],
    plugins: { jsdoc },
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
    },
    rules: {
      // Require docblocks for our top-level functions.
      "jsdoc/require-jsdoc": [
        "error",
        {
          contexts: ["FunctionDeclaration"],
          require: {
            FunctionDeclaration: true,
          },
        },
      ],

      // Validate tag presence and alignment.
      "jsdoc/require-param": "error",
      "jsdoc/require-returns": "error",
      "jsdoc/check-param-names": "error",
      "jsdoc/check-tag-names": "error",
      "jsdoc/check-types": "off",
      "jsdoc/no-undefined-types": "off",
      "jsdoc/require-param-description": "error",
      "jsdoc/require-returns-description": "error",
    },
  },
];
