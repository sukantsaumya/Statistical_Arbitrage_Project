document.addEventListener("DOMContentLoaded", function() {
    // --- Intersection Observer for Fade-in Animations ---
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.05 });

    function observeElements(selector) {
        document.querySelectorAll(selector).forEach(element => { observer.observe(element); });
    }
    observeElements('.card, .metric-card, .help-card');

    // --- Dynamic Company Info Page Logic ---
    const companyGrid = document.getElementById('companyGrid');
    const searchInput = document.getElementById('searchInput');
    const marketSelect = document.getElementById('marketSelect');
    const loadingMessage = document.getElementById('loadingMessage');

    if (companyGrid && searchInput && marketSelect) {
        let allCompanies = {};

        async function loadMarketData(marketKey) {
            loadingMessage.style.display = 'block';
            loadingMessage.textContent = `Loading data for ${marketSelect.options[marketSelect.selectedIndex].text}...`;
            companyGrid.innerHTML = '';
            searchInput.value = '';

            try {
                const response = await fetch(`/api/companies?market=${marketKey}`);
                if (!response.ok) {
                    throw new Error(`Network response was not ok: ${response.statusText}`);
                }
                const data = await response.json();

                if (Object.keys(data).length === 0) {
                    throw new Error("API returned no company data. The source (e.g., Wikipedia) may be temporarily unavailable or its structure has changed.");
                }

                allCompanies = data;
                renderCompanyGrid(allCompanies);
                loadingMessage.style.display = 'none';
            } catch (error) {
                loadingMessage.textContent = `Error: ${error.message}`;
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
                const card = document.createElement('div');
                card.className = 'company-card';
                card.style.animationDelay = `${delay}ms`;

                const logoFallback = `<div class="logo-fallback">${ticker.charAt(0)}</div>`;
                const logoImg = `<img src="${company.logo_url}" alt="${company.name} Logo" class="company-logo" onerror="this.outerHTML='${logoFallback}'">`;

                card.innerHTML = `${company.logo_url ? logoImg : logoFallback}<h3>${company.name}</h3><p class="ticker">${ticker}</p><p class="sector">${company.sector}</p>`;
                companyGrid.appendChild(card);
                observer.observe(card);
                delay += 20;
            });
        }

        searchInput.addEventListener('keyup', () => {
            const searchTerm = searchInput.value.toLowerCase();
            const filteredCompanies = Object.fromEntries(
                Object.entries(allCompanies).filter(([ticker, company]) =>
                    ticker.toLowerCase().includes(searchTerm) ||
                    (company.name && company.name.toLowerCase().includes(searchTerm)) ||
                    (company.sector && company.sector.toLowerCase().includes(searchTerm))
                )
            );
            renderCompanyGrid(filteredCompanies);
        });

        marketSelect.addEventListener('change', () => { loadMarketData(marketSelect.value); });
        loadMarketData(marketSelect.value); // Initial load
    }
});