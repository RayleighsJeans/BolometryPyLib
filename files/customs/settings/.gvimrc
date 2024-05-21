set nocompatible              " be iMproved, required
filetype off                  " required

set rtp+=$HOME\.vim\bundle\Vundle.vim
call vundle#begin('$HOME\Plugins\')

	" let Vundle manage Vundle, required
	Plugin 'VundleVim/Vundle.vim'
	Plugin 'scrooloose/syntastic'
	Plugin 'honza/vim-snippets'
	" Plugin 'SirVer/ultisnips'
	" Plugin 'Valloric/YouCompleteMe'	
	Plugin 'nvie/vim-flake8'
	Plugin 'Lokaltog/powerline'
	Plugin 'tmhedberg/SimpylFold'

	" airline stuff
	Plugin 'vim-airline/vim-airline'
	Plugin 'vim-airline/vim-airline-themes'
	
	" vimcolors
	Plugin 'lilydjwg/colorizer'
	Plugin 'dikiaap/minimalist'
	Plugin 'flazz/vim-colorschemes'
	Plugin 'jnurmine/Zenburn'
	Plugin 'altercation/vim-colors-solarized'
	
	" LaTeX Plugins
	Plugin 'vim-latex/vim-latex'
	Plugin 'lervag/vimtex'
	Plugin 'bjoernd/vim-ycm-tex'

	" Python Plugins
	Plugin 'hdima/python-syntax'
	Plugin 'python-mode/python-mode'

	" unused plugins
	Plugin 'scrooloose/nerdtree'
	Plugin 'jistr/vim-nerdtree-tabs'
	Plugin 'tpope/vim-fugitive'

call vundle#end()	
	
" General settings {{{
	filetype plugin indent on
	syntax on

	cd W:\Documents\LABGIT\IDL2PY-PORT

	set encoding=utf-8
	set fileencoding=utf-8
	
	set title
	set mouse=a
	set hlsearch
	set noautochdir

	set modeline
	set modelines=5
	set tabstop=4
	set softtabstop=0 noexpandtab
	set shiftwidth=4 smarttab
	set number
	set confirm
	
	set backspace=2
	set backspace=indent,eol,start
" }}}
" colors {{{
	if has('gui_running')
  	set background=dark
  		colorscheme solarized
	else
  		colorscheme zenburn
	endif
" }}}
" YouCompleteMe{{{
"	let g:ycm_autoclose_preview_window_after_completion=1
"	map <leader>g  :YcmCompleter GoToDefinitionElseDeclaration<CR>
" }}}
" folding {{{
	set foldmethod=indent
	set foldlevel=99
	nnoremap <space> za
" }}}
" splitting {{{
	set splitbelow
	set splitright
	" split navigations
	nnoremap <C-J> <C-W><C-J>
	nnoremap <C-K> <C-W><C-K>
	nnoremap <C-L> <C-W><C-L>
	nnoremap <C-H> <C-W><C-H>
" }}}
" Python stuff {{{
	set pythonthreedll=C:/Program\ Files/WinPython-64bit-3.5.1.2/python-3.5.1.amd64/python35.dll
	let python_highlight_all=1
	au BufNewFile,BufRead *.py
    	\ set tabstop=4
    	\ set softtabstop=4
    	\ set shiftwidth=4
    	\ set textwidth=79
    	\ set expandtab
    	\ set autoindent
    	\ set fileformat=unix
" }}}
" airline {{{
	let g:airline_powerline_fonts = 1
	let g:airline#extensions#tabline#enabled = 1
	let g:airline#extensions#tabline#left_sep = ' '
	let g:airline#extensions#tabline#left_alt_sep = '|'
	let g:airline#extensions#tabline#enabled = 1
" }}}
" vim-latex settings {{{
	set shellslash
	set grepprg=grep\ -nH\ $*
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
" nerdtree {{{
	autocmd vimenter * NERDTree
	map <F3> :NERDTreeToggle<CR>
	let NERDTreeIgnore=['\.pyc$', '\~$']
" }}}
