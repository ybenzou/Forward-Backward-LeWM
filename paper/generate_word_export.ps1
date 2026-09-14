param(
    [Parameter(Mandatory = $true)]
    [string]$AuxPath,
    [Parameter(Mandatory = $true)]
    [string]$PdftoppmPath
)

$ErrorActionPreference = 'Stop'
$paperDir = Split-Path -Parent $PSCommandPath
$repoDir = Split-Path -Parent $paperDir
$sourcePath = Join-Path $paperDir 'iclr2027_conference.tex'
$tablePath = Join-Path $paperDir 'tables\tab_k_depth_multiseed.tex'
$exportPath = Join-Path $repoDir 'word_export.tex'
$figureDir = Join-Path $repoDir 'word_export_figures'

foreach ($path in @($sourcePath, $tablePath, $AuxPath, $PdftoppmPath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required file not found: $path"
    }
}

$labels = @{}
foreach ($line in [IO.File]::ReadAllLines($AuxPath)) {
    if ($line -match '^\\newlabel\{([^}]+)\}\{\{([^}]+)\}') {
        $labels[$Matches[1]] = $Matches[2]
    }
}
if ($labels.Count -eq 0) { throw 'No LaTeX labels found in the .aux file.' }

$tex = [IO.File]::ReadAllText($sourcePath, [Text.Encoding]::UTF8)
$tex = $tex.Replace('\usepackage{iclr2027_conference,times}', '\usepackage{times,natbib}')
$tex = $tex.Replace('\title{Horizon-Aligned Scoring: A Longer View for\\', '\title{Horizon-Aligned Scoring: A Longer View for')
$tex = $tex.Replace('\input{math_commands.tex}', '')
$tex = $tex.Replace('\input{macros.tex}', '')
$tex = [regex]::Replace($tex, '(?ms)^\\author\{Yuan Ben.*?^\}', '\author{}')
$tex = $tex.Replace('\input{tables/tab_k_depth_multiseed.tex}', [IO.File]::ReadAllText($tablePath, [Text.Encoding]::UTF8))
$tex = $tex.Replace('\cmidrule(lr){4-6}', '')
$tex = $tex.Replace('\bibliography{iclr2027_conference}', '')
$tex = $tex.Replace('\bibliographystyle{iclr2027_conference}', '')
$tex = $tex.Replace('\section{Training and implementation}', '\section{Appendix A Training and implementation}')
$tex = $tex.Replace('\section{Evaluation protocol}', '\section{Appendix B Evaluation protocol}')
$tex = $tex.Replace('\section{Scoring diagnostic protocol}', '\section{Appendix C Scoring diagnostic protocol}')
$tex = $tex.Replace('\end{document}', "\section*{References}`n\bibliography{paper/iclr2027_conference}`n\bibliographystyle{paper/iclr2027_conference}`n\end{document}")

$figures = @(
    'fig_has_evaluation',
    'fig_has_training',
    'fig_iclr_multiseed',
    'fig_has_score_contrast',
    'fig_has_process'
)
New-Item -ItemType Directory -Path $figureDir -Force | Out-Null
foreach ($name in $figures) {
    $pdf = Join-Path $paperDir "figures\$name.pdf"
    if (-not (Test-Path -LiteralPath $pdf -PathType Leaf)) { throw "Figure not found: $pdf" }
    $prefix = Join-Path $figureDir $name
    & $PdftoppmPath -f 1 -l 1 -singlefile -r 300 -png $pdf $prefix
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath "$prefix.png")) {
        throw "Could not rasterize figure: $pdf"
    }
    $tex = $tex.Replace("{figures/$name.pdf}", "{word_export_figures/$name.png}")
}

$captionPrefixes = @(
    @('\caption{\textbf{Horizon-aligned', '\caption{Figure 1. \textbf{Horizon-aligned'),
    @('\caption{Local and recursive', '\caption{Figure 2. Local and recursive'),
    @('\caption{Goal reaching across', '\caption{Figure 3. Goal reaching across'),
    @('\caption{Observed endpoint-to-goal', '\caption{Figure 4. Observed endpoint-to-goal'),
    @('\caption{Paired executions', '\caption{Figure 5. Paired executions'),
    @('\caption{Success rate (\%; mean $\pm$ std over $10$', '\caption{Table 1. Success rate (\%; mean $\pm$ std over $10$'),
    @('\caption{Success rate (\%; mean $\pm$ std over ten', '\caption{Table 2. Success rate (\%; mean $\pm$ std over ten'),
    @('\caption{Success rates (\%) across three', '\caption{Table 3. Success rates (\%) across three')
)
foreach ($pair in $captionPrefixes) {
    if (-not $tex.Contains($pair[0])) { throw "Caption not found: $($pair[0])" }
    $tex = $tex.Replace($pair[0], $pair[1])
}

$algorithmStart = $tex.IndexOf('\begin{algorithm}[!htbp]')
$algorithmEnd = $tex.IndexOf('\end{algorithm}', $algorithmStart)
if ($algorithmStart -lt 0 -or $algorithmEnd -lt 0) { throw 'Algorithm block not found.' }
$algorithmEnd += '\end{algorithm}'.Length
$wordAlgorithm = @'
\paragraph{Algorithm 1. Horizon-Aligned CEM at elapsed time $e$}
Inputs: current latent and recent context; goal $z_g$; offset $o$;
predictors $P$ and $F$; block size $b$; span $H=hb$;
candidates $N$; elites $M$; updates $J$.
\begin{enumerate}
\item Set $k\gets\max((o-e-H)/b,0)$ and initialize
$(\mu_0,\sigma_0)\gets\operatorname{InitializeCEM}(h)$.
\item For $j=1,\ldots,J$, sample
$A^{(1:N)}\gets\operatorname{Sample}(\mu_{j-1},\sigma_{j-1},N)$.
For each candidate $n$, compute
$\hat z^{(n)}_{e+H}\gets P^{(h)}(z_e,A^{(n)})$,
$\tilde z^{(n)}\gets F^k(\hat z^{(n)}_{e+H})$, and
$c_n\gets\|\tilde z^{(n)}-z_g\|_2^2$.
Select $\mathcal{I}_j\gets\operatorname{Elite}(c_{1:N},M)$ and update
$(\mu_j,\sigma_j)\gets\operatorname{Moments}(\{A^{(n)}:n\in\mathcal{I}_j\})$.
\item Return $\mu_J$.
\end{enumerate}
'@
$tex = $tex.Substring(0, $algorithmStart) + $wordAlgorithm + $tex.Substring($algorithmEnd)

foreach ($key in @($labels.Keys | Where-Object { $_ -like 'eq:*' })) {
    $tex = $tex.Replace("\label{$key}", "\qquad\text{($($labels[$key]))}")
}
$tex = [regex]::Replace($tex, '\\ref\{([^}]+)\}', {
    param($match)
    $key = $match.Groups[1].Value
    if (-not $labels.ContainsKey($key)) { throw "Unresolved LaTeX reference: $key" }
    return $labels[$key]
})

$header = @'
% Word-export copy generated from paper/iclr2027_conference.tex.
% The canonical submission source remains in paper/iclr2027_conference.tex.
% Select this file as Overleaf's main document only when exporting Word.

'@
[IO.File]::WriteAllText($exportPath, $header + $tex, [Text.UTF8Encoding]::new($false))
Write-Output "Wrote $exportPath"
