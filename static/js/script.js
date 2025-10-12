// static/js/script.js (fixed & robust)
document.addEventListener("DOMContentLoaded", function() {
    // --- NAV TOGGLE (mobile) ---
    const navToggle = document.getElementById('navToggle');
    const navLinks = document.getElementById('navLinks');
    if (navToggle && navLinks) {
        navToggle.addEventListener('click', () => {
            navLinks.classList.toggle('open');
        });
    }
// Smooth Advanced toggle + caret rotation
document.addEventListener("DOMContentLoaded", function() {
  const adv = document.querySelector(".advanced");
  if (!adv) return;

  const caret = adv.querySelector(".caret");
  adv.addEventListener("toggle", () => {
    if (caret) caret.style.transform = adv.open ? "rotate(180deg)" : "rotate(0deg)";
  });
});

    // --- Form validation & UX ---
    const form = document.getElementById('backtestForm');
    const runBtn = document.getElementById('runBtn');
    const formMsg = document.getElementById('formMsg');

    if (form) {
        form.addEventListener('submit', (e) => {
            // basic client validation
            const universeEl = document.getElementById('universe');
            const universe = universeEl ? universeEl.value.trim() : '';
            if (!universe || universe.split(',').filter(Boolean).length < 2) {
                e.preventDefault();
                if (formMsg) {
                    formMsg.textContent = "Please provide at least two tickers (comma-separated).";
                    formMsg.classList.add('error');
                }
                return false;
            }

            // disable UI while submitting (safe checks)
            if (runBtn) {
                runBtn.disabled = true;
                runBtn.textContent = "Running…";
            }
            if (formMsg) {
                formMsg.textContent = "Backtest running — this may take a moment.";
                formMsg.classList.remove('error');
            }
            return true;
        });
    }

    /* -------------------------
       Company grid logic
       ------------------------- */

    const companyGrid = document.getElementById('companyGrid');
    const searchInput = document.getElementById('searchInput');
    const marketSelect = document.getElementById('marketSelect');
    const loadingMessage = document.getElementById('loadingMessage');

    // Create an IntersectionObserver for reveal animation (only if companyGrid exists)
    let cardObserver = null;
    if (companyGrid) {
        cardObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                    cardObserver.unobserve(entry.target);
                }
            });
        }, { threshold: 0.12 });
    }

    if (companyGrid && searchInput && marketSelect) {
        let allCompanies = {};

        async function loadMarketData(marketKey) {
            if (loadingMessage) {
                loadingMessage.style.display = 'block';
                loadingMessage.textContent = `Loading data for ${marketSelect.options[marketSelect.selectedIndex].text}...`;
            }
            companyGrid.innerHTML = '';
            searchInput.value = '';

            try {
                const response = await fetch(`/api/companies?market=${encodeURIComponent(marketKey)}`);
                if (!response.ok) {
                    throw new Error(`Network response was not ok: ${response.status} ${response.statusText}`);
                }
                const data = await response.json();

                if (!data || Object.keys(data).length === 0) {
                    throw new Error("API returned no company data. The source may be unavailable or schema changed.");
                }

                allCompanies = data;
                renderCompanyGrid(allCompanies);
                if (loadingMessage) loadingMessage.style.display = 'none';
            } catch (error) {
                if (loadingMessage) {
                    loadingMessage.style.display = 'block';
                    loadingMessage.textContent = `Error: ${error.message}`;
                }
                console.error("Error fetching company data:", error);
            }
        }

        function renderCompanyGrid(companies) {
            companyGrid.innerHTML = '';
            let delay = 0;
            const companyArray = Object.entries(companies);

            if (companyArray.length === 0) {
                companyGrid.innerHTML = `<p class="loading-error">No companies found matching your search.</p>`;
                return;
            }

            companyArray.forEach(([ticker, company]) => {
                // build DOM elements (safer than innerHTML concatenation)
                const card = document.createElement('div');
                card.className = 'company-card';
                card.style.animationDelay = `${delay}ms`;

                // Logo or fallback
                let logoWrapper;
                if (company && company.logo_url) {
                    const img = document.createElement('img');
                    img.className = 'company-logo';
                    img.alt = `${company.name || ticker} Logo`;
                    img.src = company.logo_url;
                    // on error replace img with fallback
                    img.addEventListener('error', () => {
                        const fallback = document.createElement('div');
                        fallback.className = 'logo-fallback';
                        fallback.textContent = ticker.charAt(0);
                        img.replaceWith(fallback);
                    });
                    logoWrapper = img;
                } else {
                    const fallback = document.createElement('div');
                    fallback.className = 'logo-fallback';
                    fallback.textContent = ticker.charAt(0);
                    logoWrapper = fallback;
                }

                // Title, ticker and sector
                const title = document.createElement('h3');
                title.textContent = company.name || ticker;

                const tSpan = document.createElement('p');
                tSpan.className = 'ticker';
                tSpan.textContent = ticker;

                const sector = document.createElement('p');
                sector.className = 'sector muted';
                sector.textContent = company.sector || '';

                // assemble
                card.appendChild(logoWrapper);
                card.appendChild(title);
                card.appendChild(tSpan);
                card.appendChild(sector);

                companyGrid.appendChild(card);

                // observe for reveal
                if (cardObserver) cardObserver.observe(card);

                delay += 20;
            });
        }

        // search filter
        searchInput.addEventListener('input', () => {
            const searchTerm = searchInput.value.trim().toLowerCase();
            if (!searchTerm) {
                renderCompanyGrid(allCompanies);
                return;
            }
            const filtered = Object.fromEntries(
                Object.entries(allCompanies).filter(([ticker, company]) => {
                    const name = company.name || '';
                    const sector = company.sector || '';
                    return ticker.toLowerCase().includes(searchTerm) ||
                           name.toLowerCase().includes(searchTerm) ||
                           sector.toLowerCase().includes(searchTerm);
                })
            );
            renderCompanyGrid(filtered);
        });

        marketSelect.addEventListener('change', () => {
            loadMarketData(marketSelect.value);
        });

        // initial load (guard if selectedIndex missing)
        const initialMarket = marketSelect.value || marketSelect.options[0].value;
        loadMarketData(initialMarket);
    }
});
// when preset_years changes, auto-fill start_date/end_date
document.getElementById('preset_years')?.addEventListener('change', function(e){
  const years = parseFloat(e.target.value);
  if (!years) return;
  const end = new Date();
  const start = new Date();
  start.setFullYear(end.getFullYear() - years);
  document.getElementById('start_date').value = start.toISOString().slice(0,10);
  document.getElementById('end_date').value = end.toISOString().slice(0,10);
});
// auto-fill start/end when preset changes, and validate date range on submit
(function(){
  const preset = document.getElementById('preset_years');
  const startDateEl = document.getElementById('start_date');
  const endDateEl = document.getElementById('end_date');
  const form = document.getElementById('backtestForm');
  const formMsg = document.getElementById('formMsg');

  function isoDate(d){ return d.toISOString().slice(0,10); }

  if (preset && startDateEl && endDateEl) {
    preset.addEventListener('change', () => {
      const years = parseFloat(preset.value);
      if (!years) return;
      const end = new Date();
      const start = new Date();
      // use approx year arithmetic (ok for UX)
      start.setFullYear(end.getFullYear() - Math.floor(years));
      startDateEl.value = isoDate(start);
      endDateEl.value = isoDate(end);
    });
  }

  if (form) {
    form.addEventListener('submit', (e) => {
      // validate date order if provided
      if (startDateEl.value && endDateEl.value) {
        const s = new Date(startDateEl.value);
        const t = new Date(endDateEl.value);
        if (s > t) {
          e.preventDefault();
          if (formMsg) {
            formMsg.textContent = "Start date must be before end date.";
            formMsg.classList.add('error');
          }
          return false;
        }
      }
      // continue with existing submit logic (your existing handler will disable button)
      return true;
    });
  }
})();
