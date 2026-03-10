package com.defooocus.browser

import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.view.inputmethod.EditorInfo
import android.widget.EditText
import android.widget.ImageButton
import androidx.activity.ComponentActivity
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.fragment.app.FragmentActivity
import androidx.viewpager2.adapter.FragmentStateAdapter
import androidx.viewpager2.widget.ViewPager2
import com.google.android.material.tabs.TabLayout
import com.google.android.material.tabs.TabLayoutMediator

private const val DEFAULT_URL = "https://colab.research.google.com/"
private const val DEFOOOOCUS_COLAB_URL = "https://colab.research.google.com/github/lllyasviel/Fooocus/blob/main/fooocus_colab.ipynb"

class MainActivity : FragmentActivity() {

    private lateinit var addressBar: EditText
    private lateinit var tabLayout: TabLayout
    private lateinit var viewPager: ViewPager2
    private lateinit var adapter: BrowserTabsAdapter
    private val tabs = mutableListOf(DEFOOOOCUS_COLAB_URL)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        addressBar = findViewById(R.id.addressBar)
        tabLayout = findViewById(R.id.tabLayout)
        viewPager = findViewById(R.id.viewPager)
        adapter = BrowserTabsAdapter(this, tabs)
        viewPager.adapter = adapter

        TabLayoutMediator(tabLayout, viewPager) { tab, position ->
            tab.text = "Tab ${position + 1}"
        }.attach()

        findViewById<ImageButton>(R.id.newTabButton).setOnClickListener {
            tabs.add(DEFOOOOCUS_COLAB_URL)
            adapter.notifyItemInserted(tabs.lastIndex)
            viewPager.setCurrentItem(tabs.lastIndex, true)
        }

        viewPager.registerOnPageChangeCallback(object : ViewPager2.OnPageChangeCallback() {
            override fun onPageSelected(position: Int) {
                addressBar.setText(tabs[position])
            }
        })

        addressBar.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_GO) {
                val input = addressBar.text?.toString().orEmpty().trim()
                val normalized = normalizeUrl(input)
                tabs[viewPager.currentItem] = normalized
                adapter.notifyItemChanged(viewPager.currentItem)
                true
            } else {
                false
            }
        }

        addressBar.setText(DEFOOOOCUS_COLAB_URL)
        startKeepAliveService()
    }

    private fun startKeepAliveService() {
        val serviceIntent = Intent(this, BrowserKeepAliveService::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            ContextCompat.startForegroundService(this, serviceIntent)
        } else {
            startService(serviceIntent)
        }
    }

    override fun onBackPressed() {
        val currentFragment = supportFragmentManager.findFragmentByTag("f${viewPager.currentItem}")
        if (currentFragment is BrowserTabFragment && currentFragment.canGoBack()) {
            currentFragment.goBack()
        } else {
            super.onBackPressed()
        }
    }

    private fun normalizeUrl(input: String): String {
        if (input.isBlank()) return DEFAULT_URL
        return if (input.startsWith("http://") || input.startsWith("https://")) input else "https://$input"
    }
}

class BrowserTabsAdapter(
    activity: FragmentActivity,
    private val tabs: List<String>
) : FragmentStateAdapter(activity) {

    override fun getItemCount(): Int = tabs.size

    override fun createFragment(position: Int): Fragment = BrowserTabFragment.newInstance(tabs[position])
}
