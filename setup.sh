mkdir -p ~/.streamlit/


echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
\n\
[theme]\n\
base='light'\n\
primaryColor='#7a9ae5'\n\
" > ~/.streamlit/config.toml

