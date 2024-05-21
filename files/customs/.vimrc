set nocompatible              " be iMproved, required
filetype off                  " required

set rtp+=$HOME\.vim\bundle\Vundle.vim
call vundle#begin('$HOME\Plugins\')

	" let Vundle manage Vundle, required
	Plugin 'VundleVim/Vundle.vim'

	" My Bundles here:
	Plugin 'scrooloose/syntastic'
	
	" Snippets are separated from the engine. Add this if you want them:
	Plugin 'honza/vim-snippets'

	" airline stuff
	Plugin 'vim-airline/vim-airline'
	Plugin 'vim-airline/vim-airline-themes'

	" vim tpope vim plugins
	Plugin 'tpope/vim-surround'
	Plugin 'tpope/vim-rails'
	Plugin 'tpope/vim-haml'
	Plugin 'tpope/vim-endwise'
	Plugin 'tpope/vim-ragtag'
	Plugin 'tpope/vim-markdown'
	Plugin 'tpope/vim-unimpaired'
	
	" vimcolors
	Plugin 'lilydjwg/colorizer'
	Plugin 'dikiaap/minimalist'
	Plugin 'flazz/vim-colorschemes'
	
	" LaTeX Plugins
	Plugin 'vim-latex/vim-latex'
	Plugin 'bjoernd/vim-ycm-tex'
	
	" Plugin 'vim-jp/cpp-vim'
	Plugin 'rhysd/vim-clang-format'
	Plugin 'realincubus/vim-clang-refactor'
	Plugin 'octol/vim-cpp-enhanced-highlight'
	
	" unused plugins
	Plugin 'ervandew/supertab'
	Plugin 'scrooloose/nerdtree'
	Plugin 'tomtom/tlib_vim'
	Plugin 'tomtom/tcomment_vim'
	Plugin 'tomtom/tselectbuffer_vim'
	Plugin 'Townk/vim-autoclose'
	Plugin 'scrooloose/nerdcommenter'

call vundle#end()	
	
" General settings {{{
	filetype plugin indent on
	syntax on
	
	set encoding=utf-8
	set fileencoding=utf-8
	
	set title
	set mouse=a
	set hlsearch
	set noautochdir

	set modeline          " enable modelines
	set modelines=5
	set tabstop=2
	set softtabstop=0 noexpandtab
	set shiftwidth=2 smarttab
	set number            " enable line numbers
	set ruler             " enable something
	set hidden            " buffer switching should be quick
	set confirm           " ask instead of just print errors
	set lazyredraw        " don't redraw while executing macros

	colorscheme Chasing_Logic
" }}}
" airline {{{
	let g:airline_powerline_fonts = 1
	let g:airline#extensions#tabline#enabled = 1
	let g:airline#extensions#tabline#left_sep = ' '
	let g:airline#extensions#tabline#left_alt_sep = '|'
	let g:airline#extensions#tabline#enabled = 1
" }}}
" vim-latex settings {{{
	" IMPORTANT: win32 users will need to have 'shellslash' set so that latex
	" can be called correctly.
	set shellslash
	" IMPORTANT: grep will sometimes skip displaying the file name if you
	" search in a singe file. This will confuse Latex-Suite. Set your grep
	" program to always generate a file-name.
	set grepprg=grep\ -nH\ $*
	" The following changes the default filetype back to 'tex':
	let g:tex_flavor='latex'
" }}}
" vim-ycm-tex {{{
	let g:ycm_semantic_triggers = { 'tex'  : ['\ref{','\cite{'], }
" }}}
" syntastic {{{
	set statusline+=%#warningmsg#
	set statusline+=%{SyntasticStatuslineFlag()}
	set statusline+=%*

	let g:syntastic_always_populate_loc_list = 1
	let g:syntastic_auto_loc_list = 1
	let g:syntastic_check_on_open = 1
	let g:syntastic_check_on_wq = 0
" }}}
" ycm clang {{{
	let g:ycm_server_python_interpreter = '/usr/bin/python2'
	let g:clang_c_options = '-std=gnu11'
	let g:clang_cpp_options = '-std=c++11 -stdlib=libc++'
	let g:ycm_confirm_extra_conf = 0 

	let g:ycm_key_list_select_completion = ['<C-n>', '<Down>']
	let g:ycm_key_list_previous_completion = ['<C-p>', '<Up>']
	let g:ycm_server_log_level = "debug"
	let g:SuperTabDefaultCompletionType = '<C-n>'
" }}}
" vim clang format {{{
	let g:clang_format#style_options = { 
		\ "AccessModifierOffset" : -4,
		\ "AllowShortIfStatementsOnASingleLine" : "true",
		\ "AlwaysBreakTemplateDeclarations" : "true",
		\ "Standard" : "C++11"}
	
	" map to <Leader>cf in C++ code
	autocmd FileType c,cpp,objc nnoremap <buffer><Leader>cf :<C-u>ClangFormat<CR>
	autocmd FileType c,cpp,objc vnoremap <buffer><Leader>cf :ClangFormat<CR>

	" if you install vim-operator-user
	autocmd FileType c,cpp,objc map <buffer><Leader>x <Plug>(operator-clang-format)

	" Toggle auto formatting:
	nmap <Leader>C :ClangFormatAutoToggle<CR>
" }}}
" vim cpp enhanced highlighting {{{
	let g:cpp_class_scope_highlight = 1
	let g:cpp_member_variable_highlight = 1
	let g:cpp_class_decl_highlight = 1
	let g:cpp_experimental_simple_template_highlight = 1
	let g:cpp_experimental_template_highlight = 1
	let g:cpp_concepts_highlight = 1
	let g:cpp_no_function_highlight = 1
" }}}
" ultisnips {{{
	" better key bindings for UltiSnipsExpandTrigger
	" Trigger configuration. Do not use <tab> if you use https://github.com/Valloric/YouCompleteMe.
	let g:UltiSnipsExpandTrigger="<tab>"
	let g:UltiSnipsJumpForwardTrigger="<c-b>"
	let g:UltiSnipsJumpBackwardTrigger="<c-z>"
	
	" If you want :UltiSnipsEdit to split your window.
	let g:UltiSnipsEditSplit="vertical"
" }}}
