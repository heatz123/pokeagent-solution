/**
 * PokéAgent Challenge - Main JavaScript
 * Interactive features and enhancements
 */

(function() {
    'use strict';

    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Video player enhancements
    const videos = document.querySelectorAll('video');

    videos.forEach(video => {
        // Pause other videos when one starts playing
        video.addEventListener('play', function() {
            videos.forEach(otherVideo => {
                if (otherVideo !== video) {
                    otherVideo.pause();
                }
            });
        });

        // Add error handling
        video.addEventListener('error', function() {
            console.error('Video failed to load:', video.src);
        });

        // Add loading indication
        video.addEventListener('loadstart', function() {
            this.classList.add('loading');
        });

        video.addEventListener('loadeddata', function() {
            this.classList.remove('loading');
        });
    });

    // Intersection Observer for fade-in animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe sections for animation
    document.querySelectorAll('.section').forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(20px)';
        section.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(section);
    });

    // Add fade-in class style
    const style = document.createElement('style');
    style.textContent = `
        .fade-in {
            opacity: 1 !important;
            transform: translateY(0) !important;
        }
    `;
    document.head.appendChild(style);

    // Scroll to top button (optional)
    const createScrollToTop = () => {
        const button = document.createElement('button');
        button.innerHTML = '<i class="fas fa-arrow-up"></i>';
        button.className = 'scroll-to-top';
        button.style.cssText = `
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: linear-gradient(135deg, #50C878 0%, #2E8B57 100%);
            color: white;
            border: none;
            cursor: pointer;
            display: none;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            z-index: 1000;
            transition: all 0.3s ease;
        `;

        button.addEventListener('click', () => {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });

        button.addEventListener('mouseenter', () => {
            button.style.transform = 'scale(1.1)';
        });

        button.addEventListener('mouseleave', () => {
            button.style.transform = 'scale(1)';
        });

        document.body.appendChild(button);

        // Show/hide button based on scroll position
        window.addEventListener('scroll', () => {
            if (window.pageYOffset > 300) {
                button.style.display = 'flex';
            } else {
                button.style.display = 'none';
            }
        });
    };

    createScrollToTop();

    // Copy citation to clipboard
    const citationBox = document.querySelector('.citation-box');
    if (citationBox) {
        citationBox.style.cursor = 'pointer';
        citationBox.title = 'Click to copy citation';

        citationBox.addEventListener('click', () => {
            const text = citationBox.textContent;
            navigator.clipboard.writeText(text).then(() => {
                // Show feedback
                const feedback = document.createElement('div');
                feedback.textContent = 'Citation copied to clipboard!';
                feedback.style.cssText = `
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: #50C878;
                    color: white;
                    padding: 15px 25px;
                    border-radius: 8px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    z-index: 10000;
                    animation: slideIn 0.3s ease;
                `;

                document.body.appendChild(feedback);

                setTimeout(() => {
                    feedback.style.animation = 'slideOut 0.3s ease';
                    setTimeout(() => feedback.remove(), 300);
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy citation:', err);
            });
        });
    }

    // Add animations CSS
    const animStyle = document.createElement('style');
    animStyle.textContent = `
        @keyframes slideIn {
            from {
                transform: translateX(400px);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }

        @keyframes slideOut {
            from {
                transform: translateX(0);
                opacity: 1;
            }
            to {
                transform: translateX(400px);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(animStyle);

    // Performance: Lazy load videos
    const videoContainers = document.querySelectorAll('.video-container video');
    const videoObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const video = entry.target;
                if (video.dataset.src) {
                    video.src = video.dataset.src;
                    delete video.dataset.src;
                }
                videoObserver.unobserve(video);
            }
        });
    });

    videoContainers.forEach(video => {
        videoObserver.observe(video);
    });

    // Add table responsiveness helper
    const tables = document.querySelectorAll('table');
    tables.forEach(table => {
        const wrapper = document.createElement('div');
        wrapper.style.overflowX = 'auto';
        table.parentNode.insertBefore(wrapper, table);
        wrapper.appendChild(table);
    });

    console.log('PokéAgent Challenge page loaded successfully! 🎮');
})();
